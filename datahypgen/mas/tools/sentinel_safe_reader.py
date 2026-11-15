"""Read Sentinel SAFE measurement windows through byte-budgeted range requests."""
import io
import re
import struct
import zipfile
from pathlib import PurePosixPath
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import numpy as np


class RangeFile(io.RawIOBase):
    def __init__(self, size, read_range):
        super().__init__()
        self.size = size
        self.position = 0
        self.read_range = read_range

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if target < 0:
            raise ValueError("negative remote seek")
        self.position = target
        return target

    def read(self, size=-1):
        count = max(0, self.size - self.position) if size < 0 else min(size, max(0, self.size - self.position))
        if not count:
            return b""
        data = self.read_range(self.position, count)
        if len(data) != count:
            raise IOError("short remote range response")
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


class SentinelSAFEReader:
    def __init__(self, owner):
        self.owner = owner
        self.used_bytes = 0

    def _reserve(self, count):
        if self.used_bytes + count > self.owner.max_stage_bytes:
            raise ValueError("Sentinel SAFE remote byte budget exceeded")
        self.used_bytes += count

    def _open(self, href):
        owner = self.owner
        if owner.is_s3_asset_href(href):
            asset = {"href": href}
            if owner.is_protected_s3_asset(asset):
                mapped = owner._nasa_https_distribution_href(asset)
                if mapped is None or not owner.nasa_edl_token:
                    raise ValueError("protected SAFE asset requires NASA HTTPS mapping and NASA_EDL_TOKEN")
                href = mapped
            else:
                import boto3
                from botocore import UNSIGNED
                from botocore.config import Config
                client_config = Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 1})
                credentials = owner._aws_session_for_asset(asset)
                mode = owner._s3_mode_for_asset(asset)
                if mode == "private" and credentials is None:
                    raise ValueError("private SAFE asset requires AWS credentials")
                if isinstance(credentials, dict):
                    client = boto3.client("s3", aws_access_key_id=credentials["AccessKeyId"],
                                          aws_secret_access_key=credentials["SecretAccessKey"],
                                          aws_session_token=credentials.get("SessionToken"), config=client_config)
                elif credentials is not None and mode != "public":
                    client = credentials.client("s3", config=client_config)
                else:
                    client = boto3.client("s3", config=client_config.merge(Config(signature_version=UNSIGNED)))
                bucket, key = owner._s3_bucket_and_key(href)
                metadata = client.head_object(Bucket=bucket, Key=key)

                def read_range(offset, count):
                    self._reserve(count)
                    response = client.get_object(Bucket=bucket, Key=key,
                                                 Range=f"bytes={offset}-{offset + count - 1}",
                                                 IfMatch=metadata["ETag"])
                    with response["Body"] as stream:
                        return stream.read(count)

                return RangeFile(metadata["ContentLength"], read_range)
        if urlparse(href).scheme not in {"http", "https"}:
            raise ValueError("unsupported SAFE asset protocol")

        def request_range(offset, count):
            self._reserve(count)
            request = Request(href, headers={**owner.headers, "Range": f"bytes={offset}-{offset + count - 1}", "Accept-Encoding": "identity"})
            with urlopen(request, timeout=30) as response:
                expected = f"bytes {offset}-{offset + count - 1}/"
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(expected):
                    raise ValueError("SAFE server must honor byte Range requests; full download refused")
                return response.read(count), int(content_range.split("/")[1])

        _, size = request_range(0, 1)

        def read_range(offset, count):
            data, current_size = request_range(offset, count)
            if current_size != size:
                raise IOError("remote SAFE asset changed during read")
            return data

        return RangeFile(size, read_range)

    @staticmethod
    def _select(names, asset):
        candidates = sorted(name for name in names if re.search(r"(?:^|/)measurement/[^/]+\.tiff?$", name, re.I))
        polarization = asset.get("polarization")
        swath = asset.get("swath")
        if polarization:
            candidates = [name for name in candidates if f"-{str(polarization).lower()}-" in name.lower()]
        if swath:
            candidates = [name for name in candidates if f"-{str(swath).lower()}-" in name.lower()]
        if not candidates:
            raise ValueError("SAFE contains no matching measurement TIFF")
        return candidates[0]

    def read(self, asset, band, max_pixels, window=None):
        href = self.owner.asset_href(asset)
        if href.lower().split("?", 1)[0].endswith(".zip"):
            with self._open(href) as remote, zipfile.ZipFile(remote) as archive:
                member = self._select(archive.namelist(), asset)
                info = archive.getinfo(member)
                if info.flag_bits & 1:
                    raise ValueError("encrypted SAFE ZIP member is unsupported")
                # DEFLATE random seeks can decompress an entire multi-GB member.
                # Stored TIFF members support true offset-based window reads.
                if info.compress_type != zipfile.ZIP_STORED:
                    raise ValueError("compressed SAFE measurement cannot guarantee bounded random access; stored ZIP member or extracted SAFE required")
                remote.seek(info.header_offset)
                header = remote.read(30)
                if header[:4] != b"PK\x03\x04":
                    raise ValueError("invalid SAFE ZIP local header")
                name_length, extra_length = struct.unpack_from("<HH", header, 26)
                offset = info.header_offset + 30 + name_length + extra_length

                def opener(path, mode="rb"):
                    if str(path) != member:
                        raise FileNotFoundError(path)
                    return RangeFile(info.file_size, lambda start, count: remote.read_range(offset + start, count))

                return self._window(member, opener, band, max_pixels, window)
        manifest_url = href if href.lower().endswith("manifest.safe") else href.rstrip("/") + "/manifest.safe"
        with self._open(manifest_url) as stream:
            if stream.size > min(self.owner.max_stage_bytes, 4 * 1024 * 1024):
                raise ValueError("SAFE manifest exceeds metadata budget")
            root = ElementTree.fromstring(stream.read())
        names = [node.attrib["href"].removeprefix("./") for node in root.iter()
                 if node.tag.rsplit("}", 1)[-1] == "fileLocation" and "href" in node.attrib]
        member = self._select(names, asset)
        path = PurePosixPath(member)
        if path.is_absolute() or ".." in path.parts or urlparse(member).scheme:
            raise ValueError("unsafe SAFE member path")
        measurement_url = manifest_url.rsplit("/", 1)[0] + "/" + member

        def opener(path, mode="rb"):
            if str(path) != member:
                raise FileNotFoundError(path)
            return self._open(measurement_url)

        return self._window(member, opener, band, max_pixels, window)

    def _window(self, member, opener, band, max_pixels, window):
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(member, opener=opener) as src:
            if window is None:
                side = max(1, int(np.sqrt(max_pixels)))
                width, height = min(side, src.width), min(side, src.height)
                window = Window((src.width - width) // 2, (src.height - height) // 2, width, height)
            if window.width * window.height > max_pixels:
                raise ValueError("SAFE window exceeds pixel budget")
            values = src.read(band, window=window, masked=True)
            complex_data = np.iscomplexobj(values)
            data = np.abs(values).astype(float) if complex_data else values.astype(float)
            data = data.filled(np.nan)
            return data, None, {
                "reader": "sentinel_safe_range_window", "measurement_member": member,
                "sample_semantics": "complex_amplitude" if complex_data else "measurement_value",
                "phase_statistics_computed": False, "remote_bytes_read": self.used_bytes,
                "window": [window.col_off, window.row_off, window.width, window.height],
                "dtype": str(src.dtypes[band - 1]), "crs": str(src.crs),
            }
