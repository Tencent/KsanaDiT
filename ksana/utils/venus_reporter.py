import atexit
import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum

from . import venus_env
from .logger import log


class MonitorReportKey:
    GENERATE_SUCCESS = "ksanadit_generate_success"
    GENERATE_FAIL = "ksanadit_generate_fail"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


class MonitorReportConfig:
    QUEUE_MAX_SIZE = 512
    DEFAULT_FUNC_NAME = "KsanaDiT"
    THREAD_NAME = "KSANA_MONITOR_REPORT_THREAD"
    ERROR_MSG_MAX_LEN = 1000
    MS_PER_SEC = 1000


_TASK_INFO_INCLUDED_KEYS = frozenset(
    [
        "request_id",
        "pipeline_key",
        "model_key",
        "prompt_len",
        "negative_prompt_len",
        "positive",
        "negative",
        "sample_config",
        "runtime_config",
        "elapsed_ms",
        "error_type",
        "error_msg",
    ]
)


class _JsonableEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Enum):
            return o.value if isinstance(o.value, (str, int, float, bool)) else str(o.value)
        if is_dataclass(o):
            return asdict(o)
        if hasattr(o, "shape"):
            return list(o.shape)
        return str(o)


def _jsonable(x):
    return json.loads(json.dumps(x, cls=_JsonableEncoder))


class VenusOpenApiReporter:
    def __init__(self):
        self.report_url = venus_env.VENUS_REPORT_URL
        self.ak = venus_env.VENUS_OPENAPI_SECRET_ID
        self.sk = venus_env.VENUS_OPENAPI_SECRET_KEY
        self._api_client = None
        self._init_error = None
        if not self.report_url or not self.ak or not self.sk:
            self._init_error = (
                "missing ENV_VENUS_REPORT_URL, ENV_VENUS_OPENAPI_SECRET_ID or ENV_VENUS_OPENAPI_SECRET_KEY"
            )
            log.warning(f"venus openapi monitor report init failed: {self._init_error}")
        else:
            try:
                from venus_api_base.venus_openapi import PyVenusOpenApi

                self._api_client = PyVenusOpenApi(self.ak, self.sk)
                log.info("venus openapi monitor report initialized successfully")
            except Exception:  # pylint: disable=broad-except
                self._init_error = True
                log.warning(f"venus openapi monitor report init failed: {self._init_error}")

    def is_initialized(self):
        return not self._init_error

    def report(self, item):
        task_info = {k: v for k, v in item.items() if k in _TASK_INFO_INCLUDED_KEYS}
        event = item.get("event")
        status = item.get("status")
        if event == MonitorReportKey.GENERATE_SUCCESS:
            status = MonitorReportKey.SUCCESS
        elif event == MonitorReportKey.GENERATE_FAIL:
            status = MonitorReportKey.ERROR

        return {
            "app_group_id": item.get("app_group_id", 0),
            "func_name": item.get("func_name", ""),
            "rtx": item.get("rtx", ""),
            "prompt": item.get("request_id"),
            "task_info": json.dumps(task_info, ensure_ascii=False),
            "status": status,
        }

    def send(self, item):
        if self._api_client is None:
            raise RuntimeError("venus openapi monitor report client not initialized")
        payload = self.report(item)
        header = {"Content-Type": "application/json"}
        body = json.dumps(payload, ensure_ascii=True)
        self._api_client.post(self.report_url, header, body)
        request_id = item.get("request_id") if isinstance(item, dict) else None
        event = item.get("event") if isinstance(item, dict) else None
        log.info(f"venus openapi monitor report sent event={event}, request_id={request_id}")


class VenusMonitorReportQueue:
    def __init__(self, maxsize):
        self.q = queue.Queue(maxsize=maxsize)
        self._started = False
        self._lock = threading.Lock()
        self._sender = None

    def set_sender(self, sender):
        self._sender = sender

    def is_initialized(self):
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._started = True
            t = threading.Thread(target=self._loop, name=MonitorReportConfig.THREAD_NAME, daemon=True)
            t.start()

    def enqueue_nowait(self, item, *, event=None, request_id=None):
        try:
            self.q.put_nowait(item)
            log.info(f"venus monitor report queue enqueued event={event}, request_id={request_id}")
            return True
        except queue.Full:
            log.warning(f"venus monitor report queue full, dropped event={event}, request_id={request_id}")
            return False

    def flush(self):
        try:
            self.q.join()
        except Exception:  # pylint: disable=broad-except
            log.error("venus monitor report queue flush failed")

    def _loop(self):
        while True:
            item = self.q.get()
            try:
                if self._sender is not None:
                    self._sender(item)
            except Exception as e:  # pylint: disable=broad-except
                log.warning(f"venus monitor report queue send failed: {e}")
            finally:
                self.q.task_done()


class VenusMonitorReportContext:
    def __init__(self, **kwargs):
        # only in real production serving environment, VenusMonitorReportContext is enabled
        self.is_report_enabled = bool(venus_env.SUMERU_APP and venus_env.SUMERU_SERVER)
        self.extra = dict(kwargs.get("extra", {})) if isinstance(kwargs.get("extra"), dict) else {}
        if not self.is_report_enabled:
            return

        self.func_name = f"{MonitorReportConfig.DEFAULT_FUNC_NAME}_{venus_env.SUMERU_APP}_{venus_env.SUMERU_SERVER}"
        self.request_id = uuid.uuid4().hex
        self.t0 = time.perf_counter()
        pipeline_key = kwargs.get("pipeline_key")
        self.pipeline_key = pipeline_key.name if isinstance(pipeline_key, Enum) else pipeline_key
        self.model_key = kwargs.get("model_key")
        self.prompt = kwargs.get("prompt")
        self.prompt_negative = kwargs.get("prompt_negative")
        self.sample_config = kwargs.get("sample_config")
        self.runtime_config = kwargs.get("runtime_config")
        self.app_group_id = venus_env.VENUS_APP_GROUP_ID
        self.rtx = venus_env.VENUS_RTX
        self.env_flag = venus_env.VENUS_ENV_FLAG
        ensure_components_initialized()
        if not _OPENAPI_REPORTER.is_initialized():
            log.error("venus monitor report skip report due to init failure")
            self.is_report_enabled = False
            return
        _REPORT_QUEUE.is_initialized()

    def report(self, event, payload):
        if not self.is_report_enabled:
            return

        data = {
            "func_name": self.func_name,
            "app_group_id": self.app_group_id,
            "rtx": self.rtx,
            "env_flag": self.env_flag,
            "event": event,
            "ts_ms": int(time.time() * MonitorReportConfig.MS_PER_SEC),
        }

        if isinstance(payload, dict):
            for k, v in payload.items():
                data[k] = _jsonable(v)
        else:
            data["payload"] = _jsonable(payload)

        _REPORT_QUEUE.enqueue_nowait(data, event=event, request_id=data.get("request_id"))

    @staticmethod
    def build_monitor_payload(
        *,
        request_id,
        pipeline_key=None,
        model_key=None,
        prompt=None,
        prompt_negative=None,
        sample_config=None,
        runtime_config=None,
        extra=None,
    ):
        payload = {
            "request_id": request_id,
            "pipeline_key": pipeline_key,
            "model_key": model_key,
            "prompt_len": len(prompt) if isinstance(prompt, str) else 0,
            "negative_prompt_len": len(prompt_negative) if isinstance(prompt_negative, str) else 0,
            "positive": prompt or "",
            "negative": prompt_negative or "",
            "sample_config": sample_config,
            "runtime_config": runtime_config,
        }
        if isinstance(extra, dict) and extra:
            payload.update(extra)
        return _jsonable(payload)

    def report_result(self, error=None, elapsed_ms=0):
        if not self.is_report_enabled:
            return
        extra_data = {**self.extra, "elapsed_ms": elapsed_ms}
        if error:
            extra_data["error_type"] = type(error).__name__ if isinstance(error, BaseException) else str(type(error))
            extra_data["error_msg"] = str(error)[: MonitorReportConfig.ERROR_MSG_MAX_LEN]
        event = MonitorReportKey.GENERATE_FAIL if error else MonitorReportKey.GENERATE_SUCCESS
        try:
            self.report(
                event,
                self.build_monitor_payload(
                    request_id=self.request_id,
                    pipeline_key=self.pipeline_key,
                    model_key=self.model_key,
                    prompt=self.prompt,
                    prompt_negative=self.prompt_negative,
                    sample_config=self.sample_config,
                    runtime_config=self.runtime_config,
                    extra=extra_data,
                ),
            )
        except Exception as e:  # pylint: disable=broad-except
            log.error(f"venus monitor report failed to report: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (
            int((time.perf_counter() - self.t0) * MonitorReportConfig.MS_PER_SEC) if self.is_report_enabled else 0
        )
        self.report_result(error=exc_val, elapsed_ms=elapsed_ms)
        return False


_OPENAPI_REPORTER = None
_REPORT_QUEUE = None
_INIT_LOCK = threading.Lock()
_INITIALIZED = False


def ensure_components_initialized():
    global _OPENAPI_REPORTER, _REPORT_QUEUE, _INITIALIZED
    if _INITIALIZED:
        return
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        _OPENAPI_REPORTER = VenusOpenApiReporter()
        _REPORT_QUEUE = VenusMonitorReportQueue(maxsize=MonitorReportConfig.QUEUE_MAX_SIZE)
        _REPORT_QUEUE.set_sender(_OPENAPI_REPORTER.send)
        atexit.register(_REPORT_QUEUE.flush)
        _INITIALIZED = True


def resolve_model_key(pipeline_key, params):
    if pipeline_key == "local_generate":
        model_key = getattr(params.get("self"), "model_key", None)
    else:
        model_obj = params.get("model")
        model_key = getattr(model_obj, "model", None) if model_obj else None
    if model_key is None:
        return "unknown"
    return model_key.name if hasattr(model_key, "name") and not isinstance(model_key, str) else str(model_key)


def extract_report_context(pipeline_key, params):
    data = {}
    for key in ("prompt", "prompt_negative", "sample_config", "runtime_config"):
        if key in params and params[key] is not None:
            data[key] = params[key]
    if "sample_config" not in data:
        cfg = {}
        for key in ("steps", "sample_guide_scale", "low_sample_guide_scale", "sample_shift", "solver_name", "denoise"):
            if key in params and params[key] is not None:
                v = params[key]
                cfg[key] = v.value if hasattr(v, "value") and not isinstance(v, (int, float, str, bool)) else v
        if cfg:
            data["sample_config"] = cfg
    if "runtime_config" not in data:
        rt = {}
        for key in ("seed", "rope_function"):
            if key in params and params[key] is not None:
                rt[key] = params[key]
        if rt:
            data["runtime_config"] = rt
    return data


def venus_report_fn(*, pipeline_key, params, elapsed_ms=0, error=None):
    model_key = resolve_model_key(pipeline_key, params)
    ctx_data = extract_report_context(pipeline_key, params)
    ctx = VenusMonitorReportContext(
        pipeline_key=pipeline_key,
        model_key=model_key,
        **ctx_data,
    )
    ctx.report_result(error=error, elapsed_ms=elapsed_ms)


def is_available():
    return bool(venus_env.SUMERU_APP and venus_env.SUMERU_SERVER)


def create_reporter():
    if not is_available():
        return None
    return venus_report_fn
