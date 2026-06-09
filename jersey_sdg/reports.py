from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def make_contact_sheet(
    crop_root: Path,
    rows: list[dict[str, Any]],
    out_path: Path,
    thumb_w: int = 128,
    thumb_h: int = 192,
    columns: int = 8,
) -> None:
    if not rows:
        return
    font = ImageFont.load_default()
    cell_h = thumb_h + 34
    sheet_w = columns * thumb_w
    sheet_h = ((len(rows) + columns - 1) // columns) * cell_h
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(rows):
        x = (idx % columns) * thumb_w
        y = (idx // columns) * cell_h
        image_path = crop_root / row["crop_path"]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.thumbnail((thumb_w, thumb_h), Image.Resampling.BICUBIC)
            px = x + (thumb_w - img.width) // 2
            sheet.paste(img, (px, y))
        label = f"#{row['gt_jersey_number']} f{row['frame_index']}"
        draw.text((x + 2, y + thumb_h + 2), label[:22], fill="black", font=font)
        draw.text((x + 2, y + thumb_h + 16), row["play_id"], fill="black", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)


def write_reid_contact_sheets(crop_root: Path, rows: list[dict[str, Any]], out_dir: Path) -> int:
    by_identity: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_identity.setdefault(row["identity_id"], []).append(row)
    count = 0
    for identity_id, identity_rows in sorted(by_identity.items()):
        out_path = out_dir / f"{identity_id}.jpg"
        make_contact_sheet(crop_root, identity_rows, out_path)
        count += 1
    return count


def write_jersey_grid(crop_root: Path, rows: list[dict[str, Any]], out_path: Path, limit: int = 800) -> None:
    rel_root = Path("../crops")
    cards = []
    for row in rows[:limit]:
        src = rel_root / row["crop_path"]
        label = html.escape(
            f"{row['identity_id']} | gt={row['gt_jersey_number']} | "
            f"attr_visible={row['jersey_visible_attr']} | f={row['frame_index']}"
        )
        cards.append(
            "<figure>"
            f"<img src='{html.escape(str(src))}' loading='lazy'>"
            f"<figcaption>{label}</figcaption>"
            "</figure>"
        )
    body = "\n".join(cards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:sans-serif;margin:16px;background:#f6f6f6}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}"
        "figure{margin:0;background:white;padding:6px;border:1px solid #ddd}"
        "img{width:100%;height:210px;object-fit:contain;background:#eee}"
        "figcaption{font-size:11px;line-height:1.25;word-break:break-word}"
        "</style></head><body>"
        f"<h1>Jersey Trial Grid</h1><p>{len(rows)} crops; showing {min(len(rows), limit)}.</p>"
        f"<div class='grid'>{body}</div></body></html>\n"
    )


def write_prediction_grid(crop_root: Path, rows: list[dict[str, Any]], out_path: Path) -> None:
    rel_root = Path("../crops")
    cards = []
    for row in rows:
        src = rel_root / row["crop_path"]
        visible = row.get("vlm_visible")
        pred = row.get("vlm_number")
        status = row.get("vlm_status")
        match = row.get("label_match")
        css = "match" if match else "mismatch"
        caption = (
            f"{row['video_id']} f{row['frame_index']} trk{row['track_id']}<br>"
            f"GT: {row['gt_jersey_number']} | Qwen: {pred} | visible: {visible}<br>"
            f"conf: {row.get('vlm_confidence')} | status: {status}<br>"
            f"{html.escape(str(row.get('vlm_reason', row.get('vlm_error', ''))))}"
        )
        cards.append(
            f"<figure class='{css}'>"
            f"<img src='{html.escape(str(src))}' loading='lazy'>"
            f"<figcaption>{caption}</figcaption>"
            "</figure>"
        )
    body = "\n".join(cards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:sans-serif;margin:16px;background:#f5f5f5;color:#111}"
        ".grid{display:grid;grid-template-columns:repeat(4,minmax(170px,1fr));gap:12px}"
        "figure{margin:0;background:white;padding:8px;border:2px solid #bbb}"
        "figure.match{border-color:#20803d} figure.mismatch{border-color:#b3261e}"
        "img{width:100%;height:230px;object-fit:contain;background:#eee}"
        "figcaption{font-size:12px;line-height:1.3;word-break:break-word;margin-top:6px}"
        "</style></head><body>"
        f"<h1>Qwen Jersey Prediction Review</h1><p>{len(rows)} examples across "
        f"{len({row['video_id'] for row in rows})} videos.</p>"
        f"<div class='grid'>{body}</div></body></html>\n"
    )


def write_gallery_prediction_grid(crop_root: Path, rows: list[dict[str, Any]], out_path: Path) -> None:
    rel_root = Path("../crops")
    cards = []
    for row in rows:
        src = rel_root / row["crop_path"]
        visible = row.get("vlm_visible")
        pred = row.get("vlm_number")
        status = row.get("vlm_status")
        match = row.get("label_match")
        css = "match" if match else "mismatch"
        label_visible = row.get("synthetic_visible")
        caption = (
            f"{row['match_id']} / {row.get('gallery_entity_id', row['video_id'])}<br>"
            f"{row['identity_id']} f{row['frame_index']} trk{row['track_id']}<br>"
            f"GT: {row['gt_jersey_number']} | Qwen: {pred} | visible: {visible}<br>"
            f"label_visible: {label_visible} | conf: {row.get('vlm_confidence')} | status: {status}<br>"
            f"{html.escape(str(row.get('vlm_reason', row.get('vlm_error', ''))))}"
        )
        cards.append(
            f"<figure class='{css}'>"
            f"<img src='{html.escape(str(src))}' loading='lazy'>"
            f"<figcaption>{caption}</figcaption>"
            "</figure>"
        )
    body = "\n".join(cards)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:sans-serif;margin:16px;background:#f5f5f5;color:#111}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}"
        "figure{margin:0;background:white;padding:8px;border:2px solid #bbb}"
        "figure.match{border-color:#20803d} figure.mismatch{border-color:#b3261e}"
        "img{width:100%;height:240px;object-fit:contain;background:#eee}"
        "figcaption{font-size:12px;line-height:1.3;word-break:break-word;margin-top:6px}"
        "</style></head><body>"
        f"<h1>Gallery Jersey Prediction Review</h1><p>{len(rows)} examples across "
        f"{len({row['match_id'] for row in rows})} matches and "
        f"{len({row.get('gallery_entity_id', row['video_id']) for row in rows})} galleries.</p>"
        f"<div class='grid'>{body}</div></body></html>\n"
    )
