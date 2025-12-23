import hashlib
import os
import time

from .logger import log

try:
    import fcntl  # Unix-only
except Exception:  # pragma: no cover
    fcntl = None


PREFETCH_BLOCK_SIZE_BYTES = 32 * 1024 * 1024
PREFETCH_DIR = "/tmp/ksanadit_prefetch"

_PREFETCHED_THIS_PROCESS: set[str] = set()


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _paths_for_file(path: str) -> tuple[str, str]:
    key = hashlib.sha1(path.encode("utf-8", errors="ignore")).hexdigest()
    return os.path.join(PREFETCH_DIR, f"{key}.lock"), os.path.join(PREFETCH_DIR, f"{key}.done")


def _done_is_valid(done_path: str, ttl_sec: int) -> bool:
    if not os.path.exists(done_path):
        return False
    if ttl_sec <= 0:
        return True
    try:
        return (time.time() - os.path.getmtime(done_path)) <= ttl_sec
    except Exception:
        return False


def maybe_prefetch_file(path: str) -> None:
    if not _env_flag("KSANA_PREFETCH_WEIGHTS", default=True):
        return
    if not path or not isinstance(path, str) or not os.path.isfile(path):
        return
    if path in _PREFETCHED_THIS_PROCESS:
        return

    os.makedirs(PREFETCH_DIR, exist_ok=True)
    lock_path, done_path = _paths_for_file(path)
    ttl_sec = _env_int("KSANA_PREFETCH_DONE_TTL_SEC", 30)

    # If flock is unavailable, fall back to per-process only.
    if fcntl is None:
        _prefetch_impl(path)
        _PREFETCHED_THIS_PROCESS.add(path)
        return

    with open(lock_path, "a+") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            if _done_is_valid(done_path, ttl_sec):
                _PREFETCHED_THIS_PROCESS.add(path)
                return

            _prefetch_impl(path)
            try:
                with open(done_path, "w"):
                    pass
            except Exception:
                pass
            _PREFETCHED_THIS_PROCESS.add(path)
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def _prefetch_impl(path: str) -> None:
    buf = bytearray(PREFETCH_BLOCK_SIZE_BYTES)
    mv = memoryview(buf)
    start = time.perf_counter()
    try:
        size = os.path.getsize(path)
    except Exception:
        size = None
    with open(path, "rb", buffering=0) as fp:
        while True:
            n = fp.readinto(mv)
            if not n:
                break
    elapsed = time.perf_counter() - start
    if size is not None and elapsed > 0:
        log.debug(f"prefetch {path} {size/1024**3:.2f} GiB in {elapsed:.2f}s")
