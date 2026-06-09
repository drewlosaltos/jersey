from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .data import PlayerAnn


def expanded_bbox_xyxy(
    bbox_xywh: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    expand: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox_xywh
    pad_x = w * expand
    pad_y = h * expand
    x1 = max(0, int(round(x - pad_x)))
    y1 = max(0, int(round(y - pad_y)))
    x2 = min(frame_width, int(round(x + w + pad_x)))
    y2 = min(frame_height, int(round(y + h + pad_y)))
    if x2 <= x1:
        x2 = min(frame_width, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_height, y1 + 1)
    return x1, y1, x2, y2


def crop_rel_path(kind: str, ann: PlayerAnn) -> Path:
    number = ann.jersey_number if ann.jersey_number else "blank"
    name = (
        f"{ann.video_id}__track{ann.track_id:03d}__jersey{number}"
        f"__frame{ann.frame_index:06d}__ann{ann.ann_id:07d}.jpg"
    )
    return Path(kind) / ann.match_id / ann.identity_id / name


def materialize_crop(
    ann: PlayerAnn,
    frames_root: Path,
    crops_root: Path,
    rel_path: Path,
    expand: float,
    jpeg_quality: int,
) -> dict[str, Any]:
    frame_path = frames_root / ann.video_id / ann.frame_file
    if not frame_path.exists():
        raise FileNotFoundError(frame_path)
    x1, y1, x2, y2 = expanded_bbox_xyxy(ann.bbox_xywh, ann.frame_width, ann.frame_height, expand)
    out_path = crops_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(frame_path) as image:
        crop = image.convert("RGB").crop((x1, y1, x2, y2))
        crop.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
    return {
        "crop_path": str(rel_path),
        "source_frame_path": str(frame_path),
        "expanded_bbox_xyxy": [x1, y1, x2, y2],
        "crop_width": x2 - x1,
        "crop_height": y2 - y1,
    }


def make_vlm_image(image_path: Path, max_side: int = 768, min_side: int = 256) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if min(w, h) * scale < min_side:
        scale = min(max_side / max(w, h), min_side / min(w, h))
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    if new_size != image.size:
        image = image.resize(new_size, Image.Resampling.BICUBIC)
    return image
