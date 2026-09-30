import logging
import sys
from typing import Optional, TextIO


def printf(str: str, verbose: bool = True):
    """Calls print on given argument and then flushes
    stdout buffer to ensure printed message is displayed right away
    """
    if verbose:
        print(str, flush=True)


# ANSI color codes
COLOR_CODES = {
    logging.WARNING: "\x1b[1;33m",
    logging.ERROR: "\x1b[1;31m",
    logging.CRITICAL: "\x1b[1;31m",
    logging.INFO: "\x1b[1;32m",
    logging.DEBUG: "\x1b[1;96m",
}
RESET_CODE = "\x1b[0m"
GRAY_CODE = "\x1b[90m"


class ColorFormatter(logging.Formatter):
    def format(self, record):
        # Add color to levelname
        color_code = COLOR_CODES.get(record.levelno, "")
        if color_code:
            record.levelname = f"{color_code}{record.levelname}{RESET_CODE}"

        return super().format(record)

    def formatTime(self, record, datefmt=None):
        # First get the regular formatted time
        asctime = super().formatTime(record, datefmt)
        # Then add color
        return f"{GRAY_CODE}{asctime}{RESET_CODE}"


def setup_logging(
    level: int = logging.INFO, log_file: Optional[str] = None, log_stream: Optional[TextIO] = None
) -> None:
    """Configures logging for the application."""
    # log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_format = "%(asctime)s - %(levelname)s - %(message)s - (%(name)s) "
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))
    if log_stream:
        handlers.append(logging.StreamHandler(log_stream))

    formatter = ColorFormatter(fmt=log_format, datefmt=date_format)
    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=handlers)

    # Completely silence specific loggers that are causing duplicate messages
    for logger_name in ["cdsapi", "urllib3", "datapi", "datapi.legacy_api_client"]:
        logger = logging.getLogger(logger_name)
        # Set to highest level possible
        logger.setLevel(logging.CRITICAL)
        # Prevent message propagation to parent loggers
        logger.propagate = False
        # Remove all handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Add null handler to prevent warnings about no handlers
        logger.addHandler(logging.NullHandler())

    # Also try to silence any potential parent loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)  # Keep the user's desired level for root
