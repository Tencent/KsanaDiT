import os

KSANA_LOGGER_LEVEL = os.getenv("KSANA_LOGGER_LEVEL", "info").lower()
KSANA_MEMORY_PROFILER = os.getenv("KSANA_MEMORY_PROFILER", "").lower() in ("1", "true", "yes")
