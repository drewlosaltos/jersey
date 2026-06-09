from __future__ import annotations

import base64
import html
import json
import argparse
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont

from .crops import expanded_bbox_xyxy, make_vlm_image
from .data import PlayerAnn, load_player_annotations
from .vlm import parse_vlm_json


EXAMPLES = [
    ("000005_boisterous_quetzal_play3", 671, 4),
    ("000005_boisterous_quetzal_play1", 377, 2),
    ("000004_intrepid_iguana_play2", 186, 4),
    ("000003_haughty_salamander_play3", 492, 4),
    ("000006_yappy_chevrotain_play2", 372, 2),
]

CROP_PROMPT = (
    "Look only at this volleyball player crop. Decide whether jersey digits on the intended "
    "player are visible and legible. Ignore unrelated people if present. Return exactly one "
    "JSON object: visible boolean, number string or null, confidence number from 0 to 1, "
    "reason short string."
)

BBOX_PROMPT = (
    "The intended volleyball player is inside the GREEN bounding box. Other annotated players "
    "are marked with RED boxes. Read only the jersey on the player in the GREEN box and ignore "
    "red-boxed players, foreground occluders, and all other people. If the green-boxed player's "
    "jersey digits are not visible or not legible, return visible false and number null. Return "
    "exactly one JSON object: visible boolean, number string or null, confidence number from 0 "
    "to 1, reason short string."
)

CROP_BBOX_PROMPT = (
    "This image is a crop, but it may contain multiple volleyball players or occluders. "
    "The intended player is inside the GREEN bounding box. Other annotated players are "
    "marked with RED boxes. Read only the jersey on the green-boxed player. If that "
    "green-boxed player's jersey digits are not visible or legible, return visible false "
    "and number null. Return exactly one JSON object: visible boolean, number string or "
    "null, confidence number from 0 to 1, reason short string."
)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _encode_image(path: Path) -> str:
    image = make_vlm_image(path, max_side=1024, min_side=256)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _label_image(path: Path, prompt: str, endpoint: str, model: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image(path)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 384,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
    }
    try:
        response = requests.post(endpoint.rstrip("/") + "/chat/completions", json=payload, timeout=180)
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        parsed = parse_vlm_json(text)
        return {"status": "ok", "raw": text, **parsed}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _frame_annotations(data_root: Path, video_id: str, frame_index: int) -> list[PlayerAnn]:
    ann_path = data_root / "annotations" / "v0" / f"{video_id}.json"
    return [ann for ann in load_player_annotations(ann_path) if ann.frame_index == frame_index]


def _make_context_image(
    data_root: Path,
    target: PlayerAnn,
    frame_anns: list[PlayerAnn],
    out_path: Path,
) -> dict[str, Any]:
    frame_path = data_root / "images_full" / target.video_id / target.frame_file
    with Image.open(frame_path) as image:
        image = image.convert("RGB")
        tx1, ty1, tx2, ty2 = expanded_bbox_xyxy(
            target.bbox_xywh, target.frame_width, target.frame_height, 0.15
        )
        tw = tx2 - tx1
        th = ty2 - ty1
        # Keep enough context to show foreground occluders but not the whole court.
        cx1 = max(0, int(tx1 - 2.0 * tw))
        cy1 = max(0, int(ty1 - 1.5 * th))
        cx2 = min(target.frame_width, int(tx2 + 2.0 * tw))
        cy2 = min(target.frame_height, int(ty2 + 1.5 * th))
        context = image.crop((cx1, cy1, cx2, cy2))

    draw = ImageDraw.Draw(context)
    font = ImageFont.load_default()
    boxes = []
    for ann in frame_anns:
        x1, y1, x2, y2 = expanded_bbox_xyxy(ann.bbox_xywh, ann.frame_width, ann.frame_height, 0.03)
        ix1, iy1 = x1 - cx1, y1 - cy1
        ix2, iy2 = x2 - cx1, y2 - cy1
        if ix2 < 0 or iy2 < 0 or ix1 > context.width or iy1 > context.height:
            continue
        is_target = ann.track_id == target.track_id and ann.ann_id == target.ann_id
        color = (0, 255, 0) if is_target else (255, 0, 0)
        width = 5 if is_target else 2
        draw.rectangle([ix1, iy1, ix2, iy2], outline=color, width=width)
        label = "TARGET" if is_target else f"trk{ann.track_id}"
        draw.text((ix1 + 2, max(0, iy1 - 12)), label, fill=color, font=font)
        boxes.append({"track_id": ann.track_id, "bbox_context_xyxy": [ix1, iy1, ix2, iy2], "target": is_target})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    context.save(out_path, quality=92)
    return {"context_crop_xyxy_in_frame": [cx1, cy1, cx2, cy2], "boxes": boxes}


def _make_bbox_on_original_crop(
    trial_root: Path,
    row: dict[str, Any],
    target: PlayerAnn,
    frame_anns: list[PlayerAnn],
    out_path: Path,
) -> dict[str, Any]:
    crop_src = trial_root / "crops" / row["crop_path"]
    crop_x1, crop_y1, crop_x2, crop_y2 = row["expanded_bbox_xyxy"]
    with Image.open(crop_src) as image:
        crop = image.convert("RGB")

    draw = ImageDraw.Draw(crop)
    font = ImageFont.load_default()
    boxes = []
    for ann in frame_anns:
        x1, y1, x2, y2 = expanded_bbox_xyxy(ann.bbox_xywh, ann.frame_width, ann.frame_height, 0.03)
        ix1, iy1 = x1 - crop_x1, y1 - crop_y1
        ix2, iy2 = x2 - crop_x1, y2 - crop_y1
        if ix2 < 0 or iy2 < 0 or ix1 > crop.width or iy1 > crop.height:
            continue
        ix1 = max(0, ix1)
        iy1 = max(0, iy1)
        ix2 = min(crop.width - 1, ix2)
        iy2 = min(crop.height - 1, iy2)
        is_target = ann.track_id == target.track_id and ann.ann_id == target.ann_id
        color = (0, 255, 0) if is_target else (255, 0, 0)
        width = 4 if is_target else 2
        draw.rectangle([ix1, iy1, ix2, iy2], outline=color, width=width)
        label = "TARGET" if is_target else f"trk{ann.track_id}"
        draw.text((ix1 + 2, max(0, iy1 - 12)), label, fill=color, font=font)
        boxes.append({"track_id": ann.track_id, "bbox_crop_xyxy": [ix1, iy1, ix2, iy2], "target": is_target})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, quality=92)
    return {"crop_xyxy_in_frame": [crop_x1, crop_y1, crop_x2, crop_y2], "boxes": boxes}


def _write_html(rows: list[dict[str, Any]], out_path: Path) -> None:
    cards = []
    for row in rows:
        crop = Path("images") / row["crop_image"]
        crop_bbox = Path("images") / row["crop_bbox_image"]
        context = Path("images") / row["context_image"]
        crop_label = row["crop_label"]
        crop_bbox_label = row["crop_bbox_label"]
        bbox_label = row["bbox_label"]
        cards.append(
            "<section>"
            f"<h2>{html.escape(row['video_id'])} f{row['frame_index']} trk{row['track_id']} GT {row['gt_jersey_number']}</h2>"
            "<div class='triple'>"
            "<figure>"
            f"<img src='{html.escape(str(crop))}'>"
            "<figcaption><b>Crop-only</b><br>"
            f"visible={crop_label.get('visible')} number={html.escape(str(crop_label.get('number')))} "
            f"conf={crop_label.get('confidence')}<br>{html.escape(str(crop_label.get('reason', crop_label.get('error', ''))))}</figcaption>"
            "</figure>"
            "<figure>"
            f"<img src='{html.escape(str(crop_bbox))}'>"
            "<figcaption><b>Original crop + bbox overlay</b><br>"
            f"visible={crop_bbox_label.get('visible')} number={html.escape(str(crop_bbox_label.get('number')))} "
            f"conf={crop_bbox_label.get('confidence')}<br>{html.escape(str(crop_bbox_label.get('reason', crop_bbox_label.get('error', ''))))}</figcaption>"
            "</figure>"
            "<figure>"
            f"<img src='{html.escape(str(context))}'>"
            "<figcaption><b>Larger context + bbox overlay</b><br>"
            f"visible={bbox_label.get('visible')} number={html.escape(str(bbox_label.get('number')))} "
            f"conf={bbox_label.get('confidence')}<br>{html.escape(str(bbox_label.get('reason', bbox_label.get('error', ''))))}</figcaption>"
            "</figure>"
            "</div></section>"
        )
    out_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:sans-serif;margin:18px;background:#f6f6f6;color:#111}"
        "section{background:#fff;border:1px solid #ccc;margin:0 0 18px;padding:12px}"
        "h2{font-size:16px;margin:0 0 10px}.triple{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}"
        "figure{margin:0}img{width:100%;height:360px;object-fit:contain;background:#eee}"
        "figcaption{font-size:13px;line-height:1.35;margin-top:6px}"
        "</style></head><body>"
        "<h1>BBox Prompt Experiment</h1>"
        "<p>Compare crop-only, original-crop bbox overlay, and larger context bbox overlay prompts.</p>"
        + "\n".join(cards)
        + "</body></html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/mnt/t/data/vball/skyball")
    parser.add_argument("--trial-root", default="/mnt/t/output/jersey_sgd/trial_002")
    parser.add_argument("--out-name", default="bbox_prompt_experiment")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    trial_root = Path(args.trial_root)
    out_root = trial_root / "reports" / args.out_name
    images_dir = out_root / "images"
    endpoint = args.endpoint
    model = args.model

    candidates = _jsonl_rows(trial_root / "manifests" / "jersey_candidates.jsonl")
    by_key = {(r["video_id"], r["frame_index"], r["track_id"]): r for r in candidates}
    results = []
    for video_id, frame_index, track_id in EXAMPLES:
        row = by_key[(video_id, frame_index, track_id)]
        frame_anns = _frame_annotations(data_root, video_id, frame_index)
        target = next(ann for ann in frame_anns if ann.ann_id == row["ann_id"])
        crop_src = trial_root / "crops" / row["crop_path"]
        crop_dst = images_dir / f"{video_id}_f{frame_index}_trk{track_id}_crop.jpg"
        crop_dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(crop_src).convert("RGB").save(crop_dst, quality=92)
        crop_bbox_dst = images_dir / f"{video_id}_f{frame_index}_trk{track_id}_bbox_original_crop.jpg"
        crop_bbox_meta = _make_bbox_on_original_crop(trial_root, row, target, frame_anns, crop_bbox_dst)
        context_dst = images_dir / f"{video_id}_f{frame_index}_trk{track_id}_bbox_context.jpg"
        context_meta = _make_context_image(data_root, target, frame_anns, context_dst)
        crop_label = _label_image(crop_dst, CROP_PROMPT, endpoint, model)
        crop_bbox_label = _label_image(crop_bbox_dst, CROP_BBOX_PROMPT, endpoint, model)
        bbox_label = _label_image(context_dst, BBOX_PROMPT, endpoint, model)
        results.append(
            {
                "video_id": video_id,
                "frame_index": frame_index,
                "track_id": track_id,
                "ann_id": row["ann_id"],
                "gt_jersey_number": row["gt_jersey_number"],
                "crop_image": crop_dst.name,
                "crop_bbox_image": crop_bbox_dst.name,
                "context_image": context_dst.name,
                "crop_bbox_meta": crop_bbox_meta,
                "context_meta": context_meta,
                "crop_label": crop_label,
                "crop_bbox_label": crop_bbox_label,
                "bbox_label": bbox_label,
            }
        )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "bbox_prompt_experiment.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    _write_html(results, out_root / "bbox_prompt_experiment.html")
    print({"examples": len(results), "html": str(out_root / "bbox_prompt_experiment.html")})


if __name__ == "__main__":
    main()
