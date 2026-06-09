from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .data import load_player_annotations


EXAMPLES = [
    ("000005_boisterous_quetzal_play3", 671, 4),
    ("000005_boisterous_quetzal_play1", 377, 2),
    ("000004_intrepid_iguana_play2", 186, 4),
    ("000003_haughty_salamander_play3", 492, 4),
    ("000006_yappy_chevrotain_play2", 372, 2),
]


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _intersect(a: list[float], b: list[float]) -> list[float] | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _xywh_to_xyxy(bbox: tuple[float, float, float, float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _round_box(box: list[float]) -> list[float]:
    return [round(v, 2) for v in box]


def _rel_box(frame_box: list[float], crop_box: list[float]) -> list[float]:
    return [
        frame_box[0] - crop_box[0],
        frame_box[1] - crop_box[1],
        frame_box[2] - crop_box[0],
        frame_box[3] - crop_box[1],
    ]


def main() -> None:
    data_root = Path("/mnt/t/data/vball/skyball")
    trial_root = Path("/mnt/t/output/jersey_sgd/trial_002")
    out_root = trial_root / "reports" / "challenging_cases_crop_bboxes"
    images_dir = out_root / "images"
    overlays_dir = out_root / "overlays"
    bboxes_dir = out_root / "bboxes"
    for path in (images_dir, overlays_dir, bboxes_dir):
        path.mkdir(parents=True, exist_ok=True)

    candidates = _jsonl_rows(trial_root / "manifests" / "jersey_candidates.jsonl")
    by_key = {(r["video_id"], r["frame_index"], r["track_id"]): r for r in candidates}
    manifest = []
    font = ImageFont.load_default()

    for video_id, frame_index, track_id in EXAMPLES:
        row = by_key[(video_id, frame_index, track_id)]
        ann_path = data_root / "annotations" / "v0" / f"{video_id}.json"
        frame_anns = [ann for ann in load_player_annotations(ann_path) if ann.frame_index == frame_index]
        crop_box = [float(v) for v in row["expanded_bbox_xyxy"]]
        crop_src = trial_root / "crops" / row["crop_path"]
        stem = f"{video_id}_f{frame_index:06d}_trk{track_id:03d}_ann{row['ann_id']:07d}"
        image_name = f"{stem}.jpg"
        overlay_name = f"{stem}_overlay.jpg"
        bbox_name = f"{stem}.bboxes.json"
        image_dst = images_dir / image_name
        overlay_dst = overlays_dir / overlay_name
        bbox_dst = bboxes_dir / bbox_name
        shutil.copy2(crop_src, image_dst)

        boxes = []
        for ann in frame_anns:
            frame_box = _xywh_to_xyxy(ann.bbox_xywh)
            clipped_frame_box = _intersect(frame_box, crop_box)
            if clipped_frame_box is None:
                continue
            full_rel = _rel_box(frame_box, crop_box)
            clipped_rel = _rel_box(clipped_frame_box, crop_box)
            is_primary = ann.ann_id == row["ann_id"]
            box_area = max(0.0, (frame_box[2] - frame_box[0]) * (frame_box[3] - frame_box[1]))
            intersection_area = (clipped_frame_box[2] - clipped_frame_box[0]) * (
                clipped_frame_box[3] - clipped_frame_box[1]
            )
            boxes.append(
                {
                    "ann_id": ann.ann_id,
                    "track_id": ann.track_id,
                    "jersey_number": ann.jersey_number,
                    "is_primary": is_primary,
                    "is_target": is_primary,
                    "bbox_xyxy_in_crop": _round_box(full_rel),
                    "bbox_xyxy_in_crop_clipped": _round_box(clipped_rel),
                    "bbox_xyxy_in_frame": _round_box(frame_box),
                    "intersection_fraction_of_player_bbox": round(intersection_area / box_area, 4)
                    if box_area
                    else 0.0,
                    "category_id": ann.category_id,
                    "category_name": ann.category_name,
                    "occluded": ann.occluded,
                    "jersey_visible_attr": ann.jersey_visible_attr,
                }
            )

        boxes.sort(key=lambda b: (not b["is_primary"], b["track_id"], b["ann_id"]))
        with Image.open(image_dst) as image:
            overlay = image.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for box in boxes:
            x1, y1, x2, y2 = box["bbox_xyxy_in_crop_clipped"]
            color = (0, 255, 0) if box["is_primary"] else (255, 0, 0)
            width = 5 if box["is_primary"] else 2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            label = "PRIMARY" if box["is_primary"] else f"trk{box['track_id']}"
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color, font=font)
        overlay.save(overlay_dst, quality=92)

        payload = {
            "image_file": f"images/{image_name}",
            "overlay_file": f"overlays/{overlay_name}",
            "video_id": video_id,
            "frame_index": frame_index,
            "primary_track_id": track_id,
            "primary_ann_id": row["ann_id"],
            "primary_gt_jersey_number": row["gt_jersey_number"],
            "crop_xyxy_in_frame": _round_box(crop_box),
            "crop_width": row["crop_width"],
            "crop_height": row["crop_height"],
            "bbox_coordinate_system": "xyxy pixels relative to original crop image unless field name says frame",
            "boxes": boxes,
        }
        bbox_dst.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        manifest.append(
            {
                "video_id": video_id,
                "frame_index": frame_index,
                "primary_track_id": track_id,
                "primary_ann_id": row["ann_id"],
                "primary_gt_jersey_number": row["gt_jersey_number"],
                "image_file": f"images/{image_name}",
                "overlay_file": f"overlays/{overlay_name}",
                "bbox_file": f"bboxes/{bbox_name}",
                "num_intersecting_boxes": len(boxes),
            }
        )

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print({"examples": len(manifest), "out": str(out_root)})


if __name__ == "__main__":
    main()
