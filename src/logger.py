import logging
import os
from datetime import datetime
from pathlib import Path

# place logs folder at project root (parent of `src`)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / f"log_{datetime.now().strftime('%Y_%m_%d')}.log"

FORMAT = '%(asctime)s-%(levelname)s-%(message)s'
LEVEL = logging.INFO

def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with a file handler writing to the project logs.

    This avoids relying on `basicConfig()` (which is a no-op if logging was
    configured earlier) and ensures logs are written to a predictable location.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LEVEL)
    logger.propagate = False

    # common formatter for both handlers
    formatter = logging.Formatter(FORMAT)

    # add a file handler if none exists
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(LEVEL)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # also ensure a console handler is present so logs appear during runs
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(LEVEL)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger