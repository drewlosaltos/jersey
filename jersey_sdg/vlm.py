from __future__ import annotations

import base64
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import requests

from .crops import make_vlm_image


PROMPT = (
    "Look only at the volleyball player crop. Decide whether jersey digits are visible "
    "and legible. Do not explain. Return exactly one JSON object with keys: visible "
    "boolean, number string or null, confidence number from 0 to 1, reason short string. "
    "If unsure, use visible false and number null."
)


def parse_vlm_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        raw = match.group(0)
    data = json.loads(raw)
    visible = bool(data.get("visible", False))
    number = data.get("number")
    if number is not None:
        number = str(number).strip()
        if not number:
            number = None
    confidence = float(data.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    return {
        "visible": visible,
        "number": number,
        "confidence": confidence,
        "reason": str(data.get("reason", ""))[:240],
    }


def image_to_data_url(image_path: Path) -> str:
    image = make_vlm_image(image_path)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def label_one(
    row: dict[str, Any],
    crop_root: Path,
    endpoint: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    crop_path = crop_root / row["crop_path"]
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(crop_path)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 384,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
    }
    start = time.time()
    result = dict(row)
    try:
        response = requests.post(endpoint.rstrip("/") + "/chat/completions", json=payload, timeout=timeout)
        response.raise_for_status()
        try:
            body = response.json()
        except Exception as exc:
            raise RuntimeError(
                f"response_json_error status={response.status_code} text={response.text[:500]!r}"
            ) from exc
        text = body["choices"][0]["message"]["content"]
        parsed = parse_vlm_json(text)
        result.update(
            {
                "vlm_status": "ok",
                "vlm_visible": parsed["visible"],
                "vlm_number": parsed["number"],
                "vlm_confidence": parsed["confidence"],
                "vlm_reason": parsed["reason"],
                "vlm_raw": text,
                "vlm_latency_s": round(time.time() - start, 4),
                "label_match": parsed["visible"] and parsed["number"] == row["gt_jersey_number"],
            }
        )
    except Exception as exc:
        result.update(
            {
                "vlm_status": "error",
                "vlm_error": f"{type(exc).__name__}: {exc}",
                "vlm_latency_s": round(time.time() - start, 4),
                "label_match": False,
            }
        )
    return result


def label_batch(
    rows: list[dict[str, Any]],
    crop_root: Path,
    endpoint: str,
    model: str,
    workers: int,
    timeout: float,
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(label_one, row, crop_root, endpoint, model, timeout) for row in rows]
        for future in as_completed(futures):
            labels.append(future.result())
    return sorted(
        labels,
        key=lambda r: (
            r.get("video_id", ""),
            r.get("frame_index", -1),
            r.get("track_id", ""),
            r.get("ann_id", -1),
            r.get("crop_path", ""),
        ),
    )
