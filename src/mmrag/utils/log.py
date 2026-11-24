import logging
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_FMT, datefmt="%H:%M:%S", stream=sys.stdout)
        for noisy in ("httpx", "chromadb", "urllib3", "sentence_transformers", "matplotlib", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger
