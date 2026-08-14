
from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure(log_level: int, log_path: Path | None = None, prefix: str | None = None) -> None:
    logger = logging.getLogger(prefix)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    for filter in logger.filters:
        logger.removeFilter(filter)

    logger.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if log_path is not None:
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    noisy_libs = [
        "PIL",
        "PIL.PngImagePlugin",
        "urllib3",
        "matplotlib",
        "imageio",
    ]
    for name in noisy_libs:
        logging.getLogger(name).setLevel(logging.WARNING)
