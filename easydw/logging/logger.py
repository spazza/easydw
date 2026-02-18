"""Library-safe logger and optional configuration helper."""

import logging

_LIB_LOGGER_NAME = "easydw"
_logger = logging.getLogger(_LIB_LOGGER_NAME)
_logger.addHandler(logging.NullHandler())


def get_logger() -> logging.Logger:
    """Return the package logger.

    The returned logger is named after the package and has a `NullHandler` by
    default, so importing the library does not emit logs unless the host
    application configures logging.

    :return: Logger instance for the package
    :rtype: logging.Logger
    """
    return _logger


def configure_logger(level: int = logging.INFO) -> None:
    """Configure a basic handler for the package logger.

    This helper is intended for quick starts or scripts that want sensible
    defaults. It does nothing if the package logger already has a non-null
    handler.

    :param level: Logging level to set on the package logger
    :type level: int
    """
    if _logger.handlers and not isinstance(_logger.handlers[-1], logging.NullHandler):
        return

    log_format = "[%(levelname)s] - [%(asctime)s] - %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(log_format))
    _logger.addHandler(handler)
    _logger.setLevel(level)
