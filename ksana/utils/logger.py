import logging
import sys
import os

KSANA_LOGGER_LEVEL = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}

ksana_log_lvl = os.environ.get("KSANA_LOGGER_LEVEL", "info").lower()
ksana_log_lvl = KSANA_LOGGER_LEVEL.get(ksana_log_lvl, logging.INFO)

# TODO: consider only log on rank 0
log = logging.getLogger(__name__)
log.setLevel(ksana_log_lvl)

log.propagate = False

if not log.handlers:
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(ksana_log_lvl)
    formatter = logging.Formatter(
        "%(asctime)s |KsanaDit| %(levelname)s|%(filename)s:%(lineno)d|%(funcName)s| %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
