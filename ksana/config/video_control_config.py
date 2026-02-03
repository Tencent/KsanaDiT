from dataclasses import dataclass, field

from .wan_experimental_config import KsanaExperimentalConfig, KsanaFETAConfig, KsanaSLGConfig


@dataclass
class KsanaVideoControlConfig:
    slg: KsanaSLGConfig | None = field(default=None)
    feta: KsanaFETAConfig | None = field(default=None)
    experimental: KsanaExperimentalConfig | None = field(default=None)
