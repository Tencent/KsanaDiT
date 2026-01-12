import logging
import sys

from .env import KSANA_LOGGER_LEVEL

KSANA_LOGGER_LEVEL_DICT = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False


def reset_logging(rank_id: int = 0):
    """
    Initialize logging for the current process.
    only log on rank 0 of world_size
    """
    ksana_log_lvl = KSANA_LOGGER_LEVEL_DICT.get(KSANA_LOGGER_LEVEL, logging.INFO)
    if log.hasHandlers():
        for handler in log.handlers[:]:
            log.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter(
        f"%(asctime)s |KsanaDit|rank_{rank_id}|%(levelname)s| %(filename)s:%(lineno)d|%(funcName)s| %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(ksana_log_lvl)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
