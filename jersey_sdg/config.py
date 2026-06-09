from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrialConfig:
    data_root: Path
    output_dir: Path
    seed: int = 20260605
    max_matches: int = 3
    max_plays_per_match: int = 3
    reid_crops_per_player: int = 8
    jersey_crops_per_player: int = 20
    bbox_expand: float = 0.15
    jpeg_quality: int = 92

    @property
    def annotations_dir(self) -> Path:
        return self.data_root / "annotations" / "v0"

    @property
    def frames_dir(self) -> Path:
        return self.data_root / "images_full"

    @property
    def crops_dir(self) -> Path:
        return self.output_dir / "crops"

    @property
    def manifests_dir(self) -> Path:
        return self.output_dir / "manifests"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"
