# Tensor

TensorPool and TensorKey — the data flow mechanism for inter-node tensor passing.

## Overview

- [`TensorPool`](tensor_pool.md) — Central tensor storage, accessed via `get()`/`put()`/`peek()`
- [`TensorKey`](tensor_key.md) — Enum keys for tensor identification (no raw strings allowed)
