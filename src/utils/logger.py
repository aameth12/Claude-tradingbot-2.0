import logging
import sys
from pathlib import Path

from src.utils.config import LOGS_DIR


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOGS_DIR / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
