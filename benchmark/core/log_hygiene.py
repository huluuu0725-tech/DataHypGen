"""Shared log levels for noisy third-party libraries."""

import logging

NOISY_THIRD_PARTY_LOGGERS = ("pdfminer", "httpx", "httpcore", "urllib3")


def silence_noisy_third_party_loggers(level: int = logging.ERROR) -> None:
    """Raise the threshold of chatty third-party loggers.

    默认 ERROR：解析轨迹对使用者没有信息量，只保留会真正影响运行结果的报错；
    需要排查第三方行为时临时传 logging.WARNING。
    """
    for name in NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(level)
