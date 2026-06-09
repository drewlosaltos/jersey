from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PLAY_RE = re.compile(r"^(?P<match>.+)_play(?P<play>\d+)$")


@dataclass(frozen=True)
class PlayerAnn:
    ann_id: int
    image_id: int
    frame_file: str
    frame_index: int
    frame_width: int
    frame_height: int
    category_id: int
    category_name: str
    bbox_xywh: tuple[float, float, float, float]
    area: float
    match_id: str
    play_id: str
    video_id: str
    track_id: int
    jersey_number: str
    jersey_visible_attr: bool
    occluded: bool
    keyframe: bool

    @property
    def identity_id(self) -> str:
        number = self.jersey_number if self.jersey_number else "blank"
        return f"{self.match_id}__track{self.track_id:03d}__jersey{number}"

    @property
    def gt_number_special(self) -> bool:
        return self.jersey_number in {"", "999"}

    def to_json(self) -> dict[str, Any]:
        item = asdict(self)
        item["bbox_xywh"] = list(self.bbox_xywh)
        item["identity_id"] = self.identity_id
        item["gt_number_special"] = self.gt_number_special
        return item


def parse_video_id(stem: str) -> tuple[str, str]:
    match = PLAY_RE.match(stem)
    if not match:
        return stem, "play0"
    return match.group("match"), f"play{match.group('play')}"


def load_player_annotations(annotation_path: Path) -> list[PlayerAnn]:
    data = json.loads(annotation_path.read_text())
    stem = annotation_path.stem
    match_id, play_id = parse_video_id(stem)
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    images = {img["id"]: img for img in data.get("images", [])}
    records: list[PlayerAnn] = []

    for ann in data.get("annotations", []):
        category_id = int(ann.get("category_id", -1))
        if category_id not in (1, 2):
            continue
        image = images.get(ann.get("image_id"))
        if not image:
            continue
        attrs = ann.get("attributes", {})
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        frame_file = str(image["file_name"])
        frame_index = int(Path(frame_file).stem)
        records.append(
            PlayerAnn(
                ann_id=int(ann["id"]),
                image_id=int(ann["image_id"]),
                frame_file=frame_file,
                frame_index=frame_index,
                frame_width=int(image["width"]),
                frame_height=int(image["height"]),
                category_id=category_id,
                category_name=categories.get(category_id, str(category_id)),
                bbox_xywh=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                area=float(ann.get("area", bbox[2] * bbox[3])),
                match_id=match_id,
                play_id=play_id,
                video_id=stem,
                track_id=int(attrs.get("track_id", -1)),
                jersey_number=str(attrs.get("jersey_number", "")),
                jersey_visible_attr=bool(attrs.get("jersey_visible", False)),
                occluded=bool(attrs.get("occluded", False)),
                keyframe=bool(attrs.get("keyframe", False)),
            )
        )
    return records


def iter_annotation_paths(annotations_dir: Path) -> list[Path]:
    return sorted(annotations_dir.glob("*.json"))


def load_trial_annotations(annotations_dir: Path, max_matches: int, max_plays_per_match: int) -> list[PlayerAnn]:
    selected: list[Path] = []
    plays_by_match: dict[str, int] = {}
    for path in iter_annotation_paths(annotations_dir):
        match_id, _ = parse_video_id(path.stem)
        if match_id not in plays_by_match:
            if len(plays_by_match) >= max_matches:
                continue
            plays_by_match[match_id] = 0
        if plays_by_match[match_id] >= max_plays_per_match:
            continue
        selected.append(path)
        plays_by_match[match_id] += 1

    records: list[PlayerAnn] = []
    for path in selected:
        records.extend(load_player_annotations(path))
    return records
