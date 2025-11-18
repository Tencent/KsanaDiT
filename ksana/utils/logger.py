import logging
import sys

# TODO: consider only log on rank 0
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s |KsanaDit| %(levelname)s|%(filename)s:%(lineno)d|%(funcName)s| %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(formatter)
log.addHandler(console_handler)
