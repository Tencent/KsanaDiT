from dataclasses import dataclass, field

from ..utils.const import DEFAULT_LORA_STRENGTH


@dataclass(frozen=True)
class KsanaLoraConfig:
    path: str | None = field(default=None)
    strength: float = field(default=DEFAULT_LORA_STRENGTH)

    def __post_init__(self):
        if self.path is None:
            raise ValueError("path must be specified")
        if self.strength is None or self.strength < 0:
            raise ValueError(f"strength must be a float gt 0, got {self.strength}")
