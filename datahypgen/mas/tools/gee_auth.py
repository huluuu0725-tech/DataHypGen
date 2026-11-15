import hashlib
import json
import os
import threading
from typing import Any, Dict

# GEE value:compute 默认 read timeout 仅 10 秒，复杂投影/大数据集场景易超时。
# 统一用此值覆盖，单位秒。
_GEE_HTTP_TIMEOUT = 300
_GEE_INITIALIZE_LOCK = threading.Lock()
_GEE_INITIALIZED_KEY = None
_GEE_DEADLINE_KEY = None
_GEE_TRANSPORT_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
)


def _make_http_transport() -> Any:
    """返回带自定义超时和环境代理的 httplib2.Http 对象。

    proxy_info_from_environment() 会读取 HTTP_PROXY / HTTPS_PROXY 环境变量，
    与其他模块的 trust_env=True 行为保持一致。
    """
    try:
        import httplib2
    except ImportError:
        # httplib2 不可用时降级，让 ee.Initialize 使用默认传输
        return None
    return httplib2.Http(
        timeout=_GEE_HTTP_TIMEOUT,
        proxy_info=httplib2.proxy_info_from_environment,
    )


def _is_ee_initialized(ee: Any) -> bool:
    data = getattr(ee, "data", None)
    is_initialized = getattr(data, "is_initialized", None)
    if not callable(is_initialized):
        return False
    return bool(is_initialized())


def is_inline_credential(credential_ref: str) -> bool:
    """凭证引用是否为 JSON 内容本身（而非文件路径）。

    只认 `{` 开头这一个形态：既排除了路径，也让"想当内容用却写坏了 JSON"
    走内容分支报错，而不是被当成路径再报一句 FileNotFoundError。
    """
    return credential_ref.lstrip().startswith("{")


def _build_initialize_key(
    ee: Any,
    credential_ref: str,
    credential_env: str,
    project: str,
) -> tuple[Any, ...]:
    if credential_ref:
        if is_inline_credential(credential_ref):
            credential_identity = ("inline", hashlib.sha256(credential_ref.encode("utf-8")).hexdigest())
        else:
            credential_path = os.path.abspath(os.path.expanduser(credential_ref))
            if not os.path.isfile(credential_path):
                raise FileNotFoundError(
                    f"{credential_env} must be service-account JSON content or a path to one, "
                    f"got: {credential_ref}"
                )
            credential_stat = os.stat(credential_path)
            credential_identity = (
                credential_path,
                credential_stat.st_mtime_ns,
                credential_stat.st_size,
            )
    else:
        credential_identity = ("persistent", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

    transport_environment = tuple(
        (name, os.getenv(name)) for name in _GEE_TRANSPORT_ENV_NAMES
    )
    return (id(ee), credential_identity, project, transport_environment)


def _build_initialize_kwargs(credentials: Any, project: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"credentials": credentials}
    if project:
        kwargs["project"] = project
    http = _make_http_transport()
    if http is not None:
        kwargs["http_transport"] = http
    return kwargs


def _inline_service_account_credentials(ee: Any, content: str, credential_env: str) -> Any:
    """由环境变量里的 JSON 内容构造服务账号凭证。

    用 ``ee.oauth.SCOPES`` 而不是自带常量，保证与文件路径分支拿到完全相同的 scope。
    """
    from google.oauth2 import service_account

    try:
        # strict=False：.env 里若用双引号包裹，\n 会被解成真换行并落在 private_key
        # 字符串内部；严格模式会把这种（唯一合理的）写法判成非法控制字符。
        payload = json.loads(content, strict=False)
    except ValueError as exc:
        raise ValueError(f"{credential_env} is not valid service-account JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{credential_env} must contain a service-account JSON object")
    if not str(payload.get("client_email", "")).strip():
        raise ValueError(f"Missing client_email in {credential_env}")

    return service_account.Credentials.from_service_account_info(
        payload, scopes=ee.oauth.SCOPES
    )


def initialize_gee_from_service_account(
    ee: Any,
    credential_env: str = "GEE_API_KEY",
    project_env: str = "EARTHENGINE_PROJECT",
) -> None:
    """确保当前进程中的 Earth Engine 客户端已按指定配置初始化。

    ``credential_env`` 指向的环境变量既可以是服务账号 JSON 内容本身（以 ``{`` 开头），
    也可以是该 JSON 的文件路径。相同凭证、project 和传输环境下重复调用只检查客户端
    状态，不会重新下载 Discovery 文档。凭证内容、文件、project、代理或 CA 配置变化，
    以及 ``ee.Reset()`` 后，会执行一次新的初始化。
    """
    global _GEE_INITIALIZED_KEY, _GEE_DEADLINE_KEY

    credential_ref = str(os.getenv(credential_env, "")).strip()
    project = str(os.getenv(project_env, "")).strip()
    initialize_key = _build_initialize_key(
        ee,
        credential_ref,
        credential_env,
        project,
    )

    if _GEE_INITIALIZED_KEY == initialize_key and _is_ee_initialized(ee):
        return

    with _GEE_INITIALIZE_LOCK:
        if _GEE_INITIALIZED_KEY == initialize_key and _is_ee_initialized(ee):
            return

        if not credential_ref:
            http = _make_http_transport()
            if project:
                try:
                    init_kw = {"project": project}
                    if http is not None:
                        init_kw["http_transport"] = http
                    ee.Initialize(**init_kw)
                except TypeError:
                    ee.Initialize()
            else:
                if http is not None:
                    ee.Initialize(http_transport=http)
                else:
                    ee.Initialize()
            _GEE_INITIALIZED_KEY = initialize_key
            _GEE_DEADLINE_KEY = None
            return

        if is_inline_credential(credential_ref):
            credentials = _inline_service_account_credentials(ee, credential_ref, credential_env)
        else:
            credential_path = os.path.abspath(os.path.expanduser(credential_ref))
            with open(credential_path, "r", encoding="utf-8") as f:
                credential_payload = json.load(f)

            client_email = str(credential_payload.get("client_email", "")).strip()
            if not client_email:
                raise ValueError(
                    f"Missing client_email in service-account credential file: {credential_path}"
                )

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
            credentials = ee.ServiceAccountCredentials(client_email, credential_path)

        init_kwargs = _build_initialize_kwargs(credentials, project)

        try:
            ee.Initialize(**init_kwargs)
        except TypeError:
            # 旧版 ee 不支持 http_transport 关键字参数，降级
            if project:
                ee.Initialize(credentials, project=project)
            else:
                ee.Initialize(credentials)

        _GEE_INITIALIZED_KEY = initialize_key
        _GEE_DEADLINE_KEY = None


def set_gee_deadline(ee: Any, milliseconds: float) -> None:
    """设置 GEE 请求 deadline；相同值不重复重建 Discovery resource。"""
    global _GEE_DEADLINE_KEY

    if milliseconds < 0:
        raise ValueError("GEE deadline must be greater than or equal to zero")
    if not _is_ee_initialized(ee):
        raise RuntimeError("GEE must be initialized before setting its deadline")

    deadline_key = (id(ee), float(milliseconds))
    if _GEE_DEADLINE_KEY == deadline_key:
        return

    with _GEE_INITIALIZE_LOCK:
        if _GEE_DEADLINE_KEY == deadline_key and _is_ee_initialized(ee):
            return
        if not _is_ee_initialized(ee):
            raise RuntimeError("GEE must be initialized before setting its deadline")
        set_deadline = getattr(getattr(ee, "data", None), "setDeadline", None)
        if not callable(set_deadline):
            raise RuntimeError("Earth Engine client does not expose ee.data.setDeadline")
        set_deadline(milliseconds)
        _GEE_DEADLINE_KEY = deadline_key
