from dataclasses import dataclass, field
from ..operations import KsanaAttentionBackend


@dataclass(frozen=True)
class KsanaAttentionConfig:
    backend: KsanaAttentionBackend | None = field(default=KsanaAttentionBackend.FLASH_ATTN)

    def __post_init__(self):
        if not KsanaAttentionBackend.support(self.backend):
            raise ValueError(
                f"attention_config {self.backend} not supported in {KsanaAttentionBackend.get_supported_list()}"
            )
