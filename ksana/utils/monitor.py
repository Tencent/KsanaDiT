import inspect
import threading
import time
from functools import wraps

from . import venus_env
from .logger import log

_REPORTERS = []
_REPORTER_FACTORIES = {}
_INIT_LOCK = threading.Lock()
_INITIALIZED = False


def register_reporter_factory(name, factory):
    if not name or factory is None:
        return
    _REPORTER_FACTORIES[name] = factory


def register_builtin_reporters():
    if not (venus_env.SUMERU_APP and venus_env.SUMERU_SERVER):
        return
    try:
        from .venus_reporter import create_reporter

        register_reporter_factory("venus", create_reporter)
    except Exception as e:  # pylint: disable=broad-except
        log.warning(f"failed to load venus reporter factory: {e}")


def ensure_initialized():
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        register_builtin_reporters()
        for name, factory in _REPORTER_FACTORIES.items():
            try:
                reporter = factory()
                if reporter is not None:
                    _REPORTERS.append(reporter)
                    log.info(f"monitor reporter enabled: {name}")
            except Exception as e:  # pylint: disable=broad-except
                log.warning(f"failed to initialize reporter {name}: {e}")
        _INITIALIZED = True


def report_inner(**kwargs):
    ensure_initialized()
    if not _REPORTERS:
        return
    for reporter in _REPORTERS:
        try:
            reporter(**kwargs)
        except Exception as e:  # pylint: disable=broad-except
            log.error(f"report_inner failed: {e}")


def report(pipeline_key):
    def decorator(fn):
        sig = inspect.signature(fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                params = bound.arguments
            except TypeError:
                params = {}

            start_time = time.perf_counter()
            error = None
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                error = e
                raise
            finally:
                report_inner(
                    pipeline_key=pipeline_key,
                    params=params,
                    elapsed_ms=int((time.perf_counter() - start_time) * 1000),
                    error=error,
                )

        return wrapper

    return decorator
