# Import 风格、类型注解与 Lint 抑制

---

## Import 风格规范（方案 B）

### 规则

| 导入层级 | 写法 | 示例 |
|---------|------|------|
| 同目录（`.`） | **相对导入** | `from .base import Foo` |
| 同子包内（`..`） | **相对导入** | `from ..core.base_node import IONode` |
| 跨子包（`...` 及以上） | **绝对导入** | `from kdit.utils.factory import AdvancedFactory` |

### 判定标准

- **"同子包"** 定义：共享 `kdit/` 下同一个一级子目录。例如 `kdit/nodes/io/` 和 `kdit/nodes/core/` 同属 `nodes` 子包。
- **三级及以上相对导入（`from ...xxx`）一律禁止**，必须改为绝对导入 `from kdit.xxx`。
- `kdit/operations/` 内部的深层嵌套（如 `backends/radial_sage_attn/`）跨子目录时也使用绝对导入。

### 示例

```python
# kdit/nodes/io/diffusion_model_loader.py

# ✅ 跨子包 → 绝对导入
from kdit.config import LoraConfig, ModelConfig
from kdit.memory import PinnedMemoryManager
from kdit.models import DiffusionModel
from kdit.utils import is_file_or_dir, log

# ✅ 同子包 (nodes) → 相对导入
from ..core.base_node import IONode
from ..core.node_factory import IONodeFactory
from ..core.node_types import NodeDispatchPolicy
```

```python
# kdit/operations/attention/backends/radial_sage_attn/radial_sage_attn.py

# ✅ 跨子目录 → 绝对导入
from kdit.operations.attention.attention_op import KsanaAttentionBackendImpl

# ❌ 禁止
# from ...attention_op import KsanaAttentionBackendImpl
```

### 适用范围

本规则适用于 `kdit/` 包下所有 Python 模块，包括但不限于：
- `kdit/nodes/` — loader / encoder / decoder / generator 节点
- `kdit/operations/` — attention / linear / fuse_qkv 算子
- `kdit/adapter/comfyui/` — ComfyUI 适配层
- `kdit/models/` — 模型实现（wan / qwen）

### 自动检查

可通过以下命令验证是否存在违规的三级相对导入：

```bash
grep -rn "from \.\.\." kdit/
```

预期输出为空。

---

## 类型注解与导入规范

### 规则

项目要求 Python ≥3.10，PEP 604（`X | Y`）和 PEP 585（`list[str]`）均已原生支持，**一般不需要 `from __future__ import annotations`**。

**唯一例外**：当类方法的返回类型引用自身类（前向引用）时，仍需 `from __future__ import annotations` 使注解延迟求值。例如 `-> Pipeline`（在 `Pipeline` 类内部）、`-> PipelineDefBuilder`（在 `PipelineDefBuilder` 类内部）。

### 禁止事项

- ❌ **禁止使用 `from __future__ import annotations`** — 除非文件中存在前向引用（类方法返回自身类型），否则不需要
- ❌ **禁止使用 `typing.TYPE_CHECKING`** — 所有导入必须是普通导入，不使用 `if TYPE_CHECKING:` 保护
- ❌ **尽量避免 `from typing import`** — 优先使用 `collections.abc`（如 `Callable`, `Sequence`, `Mapping`）和内置泛型（如 `list[str]`, `dict[str, int]`）。也尽量避免使用 `Any` ，尽量避免导入 `typing`
- ❌ **避免重复导入** — 同一个模块中不要出现多条导入同一来源的语句，应合并为一条

### 示例

```python
# ❌ 禁止 — 无前向引用时不需要 future annotations
from __future__ import annotations  # 删除

# ✅ 例外 — 类方法返回自身类型（前向引用）时需要保留
from __future__ import annotations  # 保留

class PipelineDefBuilder:
    def load(self, ...) -> PipelineDefBuilder:  # 引用自身类
        ...

# ❌ 禁止 — 不使用 TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..foo import Bar  # 改为普通导入
```

```python
# ✅ Python 3.10+ 原生支持联合类型和内置泛型
@dataclass(frozen=True)
class SampleConfig:
    steps: int | None = None
    cfg_scale: float | tuple[float, float] | None = None
```

```python
# ✅ 优先使用 collections.abc 而非 typing
from collections.abc import Callable, Sequence  # ✅
# from typing import Callable, Sequence  # ❌ 避免

```

---

## Lint 抑制注释规范

### 规则

对于**必要但未直接使用**的 import 和变量，必须添加对应的 lint 抑制注释，避免 CI 或 IDE 误报。

| 场景 | 抑制注释 | 说明 |
|------|---------|------|
| 必要但未使用的 import | `# noqa: F401  # pylint: disable=unused-import` | 如 `__init__.py` 中的 re-export、side-effect import、`TYPE_CHECKING` 块外的前向引用等 |
| 必要但未使用的变量 | `# noqa: F841  # pylint: disable=unused-variable` | 如从环境变量读取但仅用于触发副作用的变量、解构赋值中的占位变量等 |

对于如果需要因为方便都直接使用`import *`的时候，需要加上 `# noqa: F403`

### 示例

```python
# ✅ __init__.py 中 re-export（import 了但本文件未直接使用）
from .pipeline import Pipeline  # noqa: F401  # pylint: disable=unused-import
from .pipeline_def import PipelineDef  # noqa: F401  # pylint: disable=unused-import

# ✅ side-effect import（导入时触发注册逻辑，本文件不直接引用）
import kdit.generators.wan_generator  # noqa: F401  # pylint: disable=unused-import

# ✅ 必要但未使用的环境变量
SOME_FLAG = os.environ.get("SOME_FLAG", "0")  # noqa: F841  # pylint: disable=unused-variable

# ✅ 解构赋值中的占位变量
c1, c2, t, h, w = conv_weight.size()  # noqa: F841  # pylint: disable=unused-variable

# ✅ 导入所有内容， 通常不推荐，非必要情况还是显示import具体内容
from .defs import * # noqa: F403
```

### 判定标准

- **"必要的 import"** 指：删除后会导致功能异常的 import（如 re-export、side-effect 注册、`TYPE_CHECKING` 相关）
- **"必要的变量"** 指：删除后会导致功能异常的变量赋值（如环境变量读取触发副作用、模块级配置）
- 如果 import 或变量**确实不需要**，应该直接删除，而不是加抑制注释
- 抑制注释同时覆盖 flake8（`noqa: F401` / `noqa: F841`）和 pylint（`pylint: disable=unused-import` / `pylint: disable=unused-variable`），确保两种 linter 都不报警
- **注释顺序**：`# noqa` 在前，`# pylint` 在后，中间用两个空格分隔
