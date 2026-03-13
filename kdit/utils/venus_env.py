# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

VENUS_REPORT_URL = os.getenv("ENV_VENUS_REPORT_URL", "")
VENUS_OPENAPI_SECRET_ID = os.getenv("ENV_VENUS_OPENAPI_SECRET_ID", "")
VENUS_OPENAPI_SECRET_KEY = os.getenv("ENV_VENUS_OPENAPI_SECRET_KEY", "")
VENUS_APP_GROUP_ID = int(os.getenv("ENV_APP_GROUP_ID", "0"))
VENUS_RTX = os.getenv("ENV_RTX") or ""
VENUS_ENV_FLAG = os.getenv("ENV_FLAG", "")

SUMERU_APP = (os.getenv("SUMERU_APP") or "").strip()
SUMERU_SERVER = (os.getenv("SUMERU_SERVER") or "").strip()
