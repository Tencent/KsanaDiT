import logging
import sys
from .env import KSANA_LOGGER_LEVEL

KSANA_LOGGER_LEVEL_DICT = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}

ksana_log_lvl = KSANA_LOGGER_LEVEL_DICT.get(KSANA_LOGGER_LEVEL, logging.INFO)

log = logging.getLogger(__name__)
log.setLevel(ksana_log_lvl)
log.propagate = False


def init_logging(rank_id: int):
    """
    Initialize logging for the current process.
    only log on rank 0 of world_size
    """
    if not log.handlers:
        formatter = logging.Formatter(
            "%(asctime)s |KsanaDit| %(levelname)s|%(filename)s:%(lineno)d|%(funcName)s| %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(ksana_log_lvl)
        console_handler.setFormatter(formatter)
        log.addHandler(console_handler)

    if rank_id != 0:
        log.setLevel(logging.ERROR)
        return
