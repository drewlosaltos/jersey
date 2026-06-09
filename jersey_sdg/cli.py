from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

import requests

from .config import TrialConfig
from .crops import crop_rel_path, materialize_crop
from .data import load_trial_annotations
from .io_utils import read_jsonl, write_json, write_jsonl
from .reports import (
    write_gallery_prediction_grid,
    write_jersey_grid,
    write_prediction_grid,
    write_reid_contact_sheets,
)
from .sampling import SampledAnn, sample_trial
from .vlm import label_batch


def append_project_log(message: str) -> None:
    path = Path(__file__).resolve().parent / "project.txt"
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with path.open("a") as handle:
        handle.write(f"\n{stamp} - {message}\n")


def _manifest_row(sample: SampledAnn, config: TrialConfig) -> dict[str, Any]:
    ann = sample.ann
    rel_path = crop_rel_path(sample.kind, ann)
    crop_meta = materialize_crop(
        ann=ann,
        frames_root=config.frames_dir,
        crops_root=config.crops_dir,
        rel_path=rel_path,
        expand=config.bbox_expand,
        jpeg_quality=config.jpeg_quality,
    )
    row = ann.to_json()
    row.update(crop_meta)
    row.update(
        {
            "sample_kind": sample.kind,
            "sample_rank": sample.rank,
            "gt_jersey_number": ann.jersey_number,
        }
    )
    return row


def build_trial(args: argparse.Namespace) -> None:
    config = TrialConfig(
        data_root=Path(args.data_root),
        output_dir=Path(args.output),
        seed=args.seed,
        max_matches=args.max_matches,
        max_plays_per_match=args.max_plays_per_match,
        reid_crops_per_player=args.reid_per_player,
        jersey_crops_per_player=args.jersey_per_player,
    )
    records = load_trial_annotations(config.annotations_dir, config.max_matches, config.max_plays_per_match)
    reid_samples, jersey_samples = sample_trial(
        records,
        reid_per_player=config.reid_crops_per_player,
        jersey_per_player=config.jersey_crops_per_player,
        seed=config.seed,
    )

    reid_rows = [_manifest_row(sample, config) for sample in reid_samples]
    jersey_rows = [_manifest_row(sample, config) for sample in jersey_samples]
    write_jsonl(config.manifests_dir / "reid_gallery.jsonl", reid_rows)
    write_jsonl(config.manifests_dir / "jersey_candidates.jsonl", jersey_rows)
    contact_sheets = write_reid_contact_sheets(
        config.crops_dir,
        reid_rows,
        config.reports_dir / "reid_contact_sheets",
    )
    write_jersey_grid(config.crops_dir, jersey_rows, config.reports_dir / "jersey_visible_grid.html")
    summary = {
        "records_loaded": len(records),
        "reid_crops": len(reid_rows),
        "jersey_candidate_crops": len(jersey_rows),
        "identities": len({r["identity_id"] for r in reid_rows + jersey_rows}),
        "matches": len({r["match_id"] for r in reid_rows + jersey_rows}),
        "videos": len({r["video_id"] for r in reid_rows + jersey_rows}),
        "contact_sheets": contact_sheets,
        "jersey_visible_attr_count": sum(1 for r in jersey_rows if r["jersey_visible_attr"]),
        "gt_number_special_count": sum(1 for r in jersey_rows if r["gt_number_special"]),
    }
    write_json(config.reports_dir / "summary.json", summary)
    append_project_log(
        "trial crops built: "
        f"{summary['reid_crops']} reid, {summary['jersey_candidate_crops']} jersey, "
        f"{summary['identities']} identities -> {config.output_dir}"
    )
    print(summary)


def label_jerseys(args: argparse.Namespace) -> None:
    output = Path(args.output)
    rows = read_jsonl(output / "manifests" / "jersey_candidates.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    labels = label_batch(
        rows=rows,
        crop_root=output / "crops",
        endpoint=args.endpoint,
        model=args.model,
        workers=args.workers,
        timeout=args.timeout,
    )
    out_path = output / "manifests" / "jersey_vlm_labels.jsonl"
    write_jsonl(out_path, labels)
    ok = sum(1 for row in labels if row.get("vlm_status") == "ok")
    matches = sum(1 for row in labels if row.get("label_match"))
    summary_path = output / "reports" / "vlm_summary.json"
    write_json(
        summary_path,
        {
            "attempted": len(labels),
            "ok": ok,
            "errors": len(labels) - ok,
            "label_matches": matches,
            "endpoint": args.endpoint,
            "model": args.model,
        },
    )
    append_project_log(f"vlm labels: {ok}/{len(labels)} ok, {matches} matches -> {out_path}")
    print({"attempted": len(labels), "ok": ok, "matches": matches, "out": str(out_path)})


def _diverse_rows_for_video(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    rows = sorted(rows, key=lambda row: (not row["jersey_visible_attr"], -row["area"], row["frame_index"]))
    visible = [row for row in rows if row["jersey_visible_attr"]]
    non_visible = [row for row in rows if not row["jersey_visible_attr"]]
    pools = [visible, non_visible, rows]
    picked: list[dict[str, Any]] = []
    seen: set[int] = set()
    for pool in pools:
        if len(picked) >= count:
            break
        if not pool:
            continue
        need = count - len(picked)
        step = max(1, len(pool) // need)
        for row in pool[::step]:
            if row["ann_id"] in seen:
                continue
            picked.append(row)
            seen.add(row["ann_id"])
            if len(picked) >= count:
                break
    return sorted(picked[:count], key=lambda row: row["frame_index"])


def _gallery_rows_for_identity(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    rows = sorted(rows, key=lambda row: (row["sample_rank"], row["frame_index"], row["ann_id"]))
    if count == 1:
        return [rows[0]]
    step = (len(rows) - 1) / (count - 1)
    picked = [rows[round(idx * step)] for idx in range(count)]
    return sorted(picked, key=lambda row: row["frame_index"])


def _sample_gallery_review_rows(
    rows: list[dict[str, Any]],
    matches: int,
    identities_per_match: int,
    images_per_identity: int,
) -> list[dict[str, Any]]:
    rows = [row for row in rows if not row.get("is_unknown_number")]
    match_ids = sorted({row["match_id"] for row in rows})[:matches]
    selected: list[dict[str, Any]] = []
    for match_id in match_ids:
        match_rows = [row for row in rows if row["match_id"] == match_id]
        identities = sorted({row["identity_id"] for row in match_rows})[:identities_per_match]
        for identity_id in identities:
            identity_rows = [row for row in match_rows if row["identity_id"] == identity_id]
            selected.extend(_gallery_rows_for_identity(identity_rows, images_per_identity))
    return selected


def _prepare_gallery_review_rows(rows: list[dict[str, Any]], gallery_crops: Path, out_crops: Path) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        src = gallery_crops / row["crop_path"]
        dst = out_crops / row["crop_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        prepared_row = dict(row)
        prepared_row["gt_jersey_number"] = str(row["jersey_number"])
        prepared_row["jersey_visible_attr"] = bool(row.get("jersey_visible", False))
        prepared.append(prepared_row)
    return prepared


def _prepare_gallery_label_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    prepared = dict(row)
    prepared["split"] = split
    prepared["gt_jersey_number"] = str(row["jersey_number"])
    prepared["jersey_visible_attr"] = bool(row.get("jersey_visible", False))
    return prepared


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def label_gallery_split(args: argparse.Namespace) -> None:
    gallery_split = Path(args.gallery_root) / args.split
    gallery_crops = gallery_split / "crops"
    source_manifest = gallery_split / "manifests" / "gallery_samples.jsonl"
    output = Path(args.output)
    out_jsonl = output / "manifests" / f"gallery_{args.split}_visibility_labels.jsonl"
    rows = [
        _prepare_gallery_label_row(row, args.split)
        for row in read_jsonl(source_manifest)
        if args.include_unknown or not row.get("is_unknown_number")
    ]
    if args.limit:
        rows = rows[: args.limit]

    done_paths = set()
    if out_jsonl.exists() and not args.overwrite:
        done_paths = {row["crop_path"] for row in read_jsonl(out_jsonl)}
    elif args.overwrite and out_jsonl.exists():
        out_jsonl.unlink()

    pending = [row for row in rows if row["crop_path"] not in done_paths]
    total = len(rows)
    try:
        response = requests.get(args.endpoint.rstrip("/") + "/models", timeout=10)
        response.raise_for_status()
    except Exception as exc:
        raise SystemExit(f"VLM endpoint is not ready: {args.endpoint} ({type(exc).__name__}: {exc})") from exc
    print(
        {
            "split": args.split,
            "total": total,
            "already_done": len(done_paths),
            "pending": len(pending),
            "out": str(out_jsonl),
        },
        flush=True,
    )

    for start in range(0, len(pending), args.chunk_size):
        chunk = pending[start : start + args.chunk_size]
        labels = label_batch(
            rows=chunk,
            crop_root=gallery_crops,
            endpoint=args.endpoint,
            model=args.model,
            workers=args.workers,
            timeout=args.timeout,
        )
        for row in labels:
            has_known_number = row.get("soccernet_label", row.get("gt_jersey_number")) != -1
            row["synthetic_visible"] = bool(has_known_number and row.get("label_match"))
            row["synthetic_visibility_rule"] = "vlm_visible_and_number_equals_gt"
        _append_jsonl(out_jsonl, labels)
        done = len(done_paths) + min(start + len(chunk), len(pending))
        ok = sum(1 for row in labels if row.get("vlm_status") == "ok")
        visible = sum(1 for row in labels if row.get("synthetic_visible"))
        print(
            {
                "split": args.split,
                "done": done,
                "total": total,
                "chunk": len(labels),
                "chunk_ok": ok,
                "chunk_visible": visible,
            },
            flush=True,
        )
        if ok == 0 and labels:
            raise SystemExit("Aborting: chunk produced 0 successful VLM labels; check the model server.")

    labels = read_jsonl(out_jsonl) if out_jsonl.exists() else []
    summary = {
        "split": args.split,
        "attempted": len(labels),
        "expected": total,
        "ok": sum(1 for row in labels if row.get("vlm_status") == "ok"),
        "errors": sum(1 for row in labels if row.get("vlm_status") != "ok"),
        "synthetic_visible": sum(1 for row in labels if row.get("synthetic_visible")),
        "matches": len({row["match_id"] for row in labels}),
        "identities": len({row["identity_id"] for row in labels}),
        "source_manifest": str(source_manifest),
        "endpoint": args.endpoint,
        "model": args.model,
    }
    summary_path = output / "reports" / f"gallery_{args.split}_visibility_summary.json"
    write_json(summary_path, summary)
    append_project_log(
        f"gallery {args.split} visibility labels: "
        f"{summary['ok']}/{summary['attempted']} ok, "
        f"{summary['synthetic_visible']} visible -> {out_jsonl}"
    )
    print({"summary": summary, "labels": str(out_jsonl), "summary_path": str(summary_path)}, flush=True)


def label_manifest(args: argparse.Namespace) -> None:
    source_manifest = Path(args.source_manifest)
    crop_root = Path(args.crop_root)
    out_jsonl = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    rows = read_jsonl(source_manifest)
    if args.limit:
        rows = rows[: args.limit]

    done_paths = set()
    if out_jsonl.exists() and not args.overwrite:
        done_paths = {row["crop_path"] for row in read_jsonl(out_jsonl)}
    elif args.overwrite and out_jsonl.exists():
        out_jsonl.unlink()

    pending = [row for row in rows if row["crop_path"] not in done_paths]
    try:
        response = requests.get(args.endpoint.rstrip("/") + "/models", timeout=10)
        response.raise_for_status()
    except Exception as exc:
        raise SystemExit(f"VLM endpoint is not ready: {args.endpoint} ({type(exc).__name__}: {exc})") from exc
    print(
        {
            "source_manifest": str(source_manifest),
            "total": len(rows),
            "already_done": len(done_paths),
            "pending": len(pending),
            "out": str(out_jsonl),
        },
        flush=True,
    )

    for start in range(0, len(pending), args.chunk_size):
        chunk = pending[start : start + args.chunk_size]
        labels = label_batch(
            rows=chunk,
            crop_root=crop_root,
            endpoint=args.endpoint,
            model=args.model,
            workers=args.workers,
            timeout=args.timeout,
        )
        for row in labels:
            row["synthetic_visible"] = bool(row.get("label_match"))
            row["synthetic_visibility_rule"] = "vlm_visible_and_number_equals_gt"
        _append_jsonl(out_jsonl, labels)
        done = len(done_paths) + min(start + len(chunk), len(pending))
        ok = sum(1 for row in labels if row.get("vlm_status") == "ok")
        visible = sum(1 for row in labels if row.get("synthetic_visible"))
        print(
            {
                "done": done,
                "total": len(rows),
                "chunk": len(labels),
                "chunk_ok": ok,
                "chunk_visible": visible,
            },
            flush=True,
        )
        if ok == 0 and labels:
            raise SystemExit("Aborting: chunk produced 0 successful VLM labels; check the model server.")

    labels = read_jsonl(out_jsonl) if out_jsonl.exists() else []
    summary = {
        "attempted": len(labels),
        "expected": len(rows),
        "ok": sum(1 for row in labels if row.get("vlm_status") == "ok"),
        "errors": sum(1 for row in labels if row.get("vlm_status") != "ok"),
        "synthetic_visible": sum(1 for row in labels if row.get("synthetic_visible")),
        "source_manifest": str(source_manifest),
        "crop_root": str(crop_root),
        "endpoint": args.endpoint,
        "model": args.model,
    }
    write_json(summary_path, summary)
    print({"summary": summary, "labels": str(out_jsonl), "summary_path": str(summary_path)}, flush=True)


def build_review(args: argparse.Namespace) -> None:
    output = Path(args.output)
    candidates = read_jsonl(output / "manifests" / "jersey_candidates.jsonl")
    all_videos = sorted({row["video_id"] for row in candidates})
    videos = all_videos[args.video_offset : args.video_offset + args.videos]
    selected: list[dict[str, Any]] = []
    for video in videos:
        rows = [row for row in candidates if row["video_id"] == video]
        selected.extend(_diverse_rows_for_video(rows, args.per_video))
    labels = label_batch(
        rows=selected,
        crop_root=output / "crops",
        endpoint=args.endpoint,
        model=args.model,
        workers=args.workers,
        timeout=args.timeout,
    )
    out_jsonl = output / "manifests" / args.labels_name
    out_html = output / "reports" / args.html_name
    write_jsonl(out_jsonl, labels)
    write_prediction_grid(output / "crops", labels, out_html)
    ok = sum(1 for row in labels if row.get("vlm_status") == "ok")
    append_project_log(f"review grid: {ok}/{len(labels)} labels -> {out_html}")
    print({"attempted": len(labels), "ok": ok, "html": str(out_html), "labels": str(out_jsonl)})


def build_gallery_review(args: argparse.Namespace) -> None:
    gallery_split = Path(args.gallery_root) / args.split
    gallery_crops = gallery_split / "crops"
    source_manifest = gallery_split / "manifests" / "gallery_samples.jsonl"
    output = Path(args.output)
    rows = read_jsonl(source_manifest)
    selected = _sample_gallery_review_rows(
        rows,
        matches=args.matches,
        identities_per_match=args.identities_per_match,
        images_per_identity=args.images_per_identity,
    )
    if args.limit:
        selected = selected[: args.limit]
    prepared = _prepare_gallery_review_rows(selected, gallery_crops, output / "crops")
    labels = label_batch(
        rows=prepared,
        crop_root=output / "crops",
        endpoint=args.endpoint,
        model=args.model,
        workers=args.workers,
        timeout=args.timeout,
    )
    for row in labels:
        row["synthetic_visible"] = bool(row.get("label_match"))
    out_jsonl = output / "manifests" / args.labels_name
    out_html = output / "reports" / args.html_name
    write_jsonl(output / "manifests" / "gallery_review_samples.jsonl", prepared)
    write_jsonl(out_jsonl, labels)
    write_gallery_prediction_grid(output / "crops", labels, out_html)
    ok = sum(1 for row in labels if row.get("vlm_status") == "ok")
    visible = sum(1 for row in labels if row.get("synthetic_visible"))
    summary = {
        "attempted": len(labels),
        "ok": ok,
        "errors": len(labels) - ok,
        "synthetic_visible": visible,
        "matches": len({row["match_id"] for row in labels}),
        "galleries": len({row.get("gallery_entity_id", row["video_id"]) for row in labels}),
        "source_manifest": str(source_manifest),
        "endpoint": args.endpoint,
        "model": args.model,
    }
    write_json(output / "reports" / "gallery_review_summary.json", summary)
    append_project_log(f"gallery review grid: {ok}/{len(labels)} labels -> {out_html}")
    print({"attempted": len(labels), "ok": ok, "visible": visible, "html": str(out_html), "labels": str(out_jsonl)})


def main() -> None:
    parser = argparse.ArgumentParser(prog="jersey_sdg")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-trial")
    build.add_argument("--data-root", default="/mnt/t/data/vball/skyball")
    build.add_argument("--output", default="/mnt/t/output/jersey_sgd/trial_001")
    build.add_argument("--seed", type=int, default=20260605)
    build.add_argument("--max-matches", type=int, default=3)
    build.add_argument("--max-plays-per-match", type=int, default=3)
    build.add_argument("--reid-per-player", type=int, default=8)
    build.add_argument("--jersey-per-player", type=int, default=20)
    build.set_defaults(func=build_trial)

    label = sub.add_parser("label-jerseys")
    label.add_argument("--output", default="/mnt/t/output/jersey_sgd/trial_001")
    label.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    label.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    label.add_argument("--workers", type=int, default=4)
    label.add_argument("--timeout", type=float, default=120.0)
    label.add_argument("--limit", type=int, default=0)
    label.set_defaults(func=label_jerseys)

    review = sub.add_parser("build-review")
    review.add_argument("--output", default="/mnt/t/output/jersey_sgd/trial_001")
    review.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    review.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    review.add_argument("--workers", type=int, default=4)
    review.add_argument("--timeout", type=float, default=180.0)
    review.add_argument("--videos", type=int, default=4)
    review.add_argument("--video-offset", type=int, default=0)
    review.add_argument("--per-video", type=int, default=8)
    review.add_argument("--labels-name", default="review_32_vlm_labels.jsonl")
    review.add_argument("--html-name", default="review_32_grid.html")
    review.set_defaults(func=build_review)

    gallery = sub.add_parser("build-gallery-review")
    gallery.add_argument("--gallery-root", default="/mnt/t/data/vball/skyball/jersey/gallery/v0")
    gallery.add_argument("--split", default="val")
    gallery.add_argument("--output", default="/mnt/t/output/jersey_sgd/gallery_review_v0")
    gallery.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    gallery.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    gallery.add_argument("--workers", type=int, default=4)
    gallery.add_argument("--timeout", type=float, default=180.0)
    gallery.add_argument("--matches", type=int, default=3)
    gallery.add_argument("--identities-per-match", type=int, default=4)
    gallery.add_argument("--images-per-identity", type=int, default=3)
    gallery.add_argument("--limit", type=int, default=0)
    gallery.add_argument("--labels-name", default="gallery_review_vlm_labels.jsonl")
    gallery.add_argument("--html-name", default="gallery_review_grid.html")
    gallery.set_defaults(func=build_gallery_review)

    gallery_label = sub.add_parser("label-gallery-split")
    gallery_label.add_argument("--gallery-root", default="/mnt/t/data/vball/skyball/jersey/gallery/v0")
    gallery_label.add_argument("--split", required=True, choices=["train", "val", "trn"])
    gallery_label.add_argument("--output", default="/mnt/t/output/jersey_sgd/gallery_visibility_v0")
    gallery_label.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    gallery_label.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    gallery_label.add_argument("--workers", type=int, default=4)
    gallery_label.add_argument("--timeout", type=float, default=240.0)
    gallery_label.add_argument("--chunk-size", type=int, default=250)
    gallery_label.add_argument("--limit", type=int, default=0)
    gallery_label.add_argument("--include-unknown", action="store_true")
    gallery_label.add_argument("--overwrite", action="store_true")
    gallery_label.set_defaults(func=label_gallery_split)

    manifest_label = sub.add_parser("label-manifest")
    manifest_label.add_argument("--source-manifest", required=True)
    manifest_label.add_argument("--crop-root", required=True)
    manifest_label.add_argument("--output-jsonl", required=True)
    manifest_label.add_argument("--summary-json", required=True)
    manifest_label.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    manifest_label.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    manifest_label.add_argument("--workers", type=int, default=4)
    manifest_label.add_argument("--timeout", type=float, default=240.0)
    manifest_label.add_argument("--chunk-size", type=int, default=250)
    manifest_label.add_argument("--limit", type=int, default=0)
    manifest_label.add_argument("--overwrite", action="store_true")
    manifest_label.set_defaults(func=label_manifest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
