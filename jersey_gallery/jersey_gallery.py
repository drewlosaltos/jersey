from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


PLAY_RE = re.compile(r"^(?P<match>.+)_play(?P<play>\d+)$")
UNKNOWN_NUMBERS = {"", "999"}
NEAR_SIDE_CATEGORY = 1
FAR_SIDE_CATEGORY = 2


@dataclass(frozen=True)
class PlayerBox:
    ann_id: int
    image_id: int
    frame_file: str
    frame_index: int
    frame_width: int
    frame_height: int
    category_id: int
    bbox_xywh: tuple[float, float, float, float]
    area: float
    match_id: str
    play_id: str
    video_id: str
    track_id: int
    jersey_number: str
    jersey_visible: bool
    occluded: bool
    jump: bool
    keyframe: bool
    gallery_entity_id: str = ""
    stable_team_id: str = ""
    adjusted_y2: float | None = None
    max_iou: float = 0.0
    high_iou_count: int = 0
    max_other_target_coverage: float = 0.0
    other_bbox_union_frac: float = 0.0
    other_bbox_sum_frac: float = 0.0
    max_other_upper_frac_in_crop: float = 0.0
    front_of_high_iou: bool = True
    salience_reason: str = "unchecked"

    @property
    def raw_y2(self) -> float:
        return self.bbox_xywh[1] + self.bbox_xywh[3]

    @property
    def team_side(self) -> str:
        if self.category_id == NEAR_SIDE_CATEGORY:
            return "near"
        if self.category_id == FAR_SIDE_CATEGORY:
            return "far"
        return f"cat{self.category_id}"

    @property
    def identity_id(self) -> str:
        entity_id = self.gallery_entity_id or self.match_id
        team_id = self.stable_team_id or self.team_side
        return f"{entity_id}__{team_id}__jersey{self.jersey_number}"

    @property
    def is_unknown_number(self) -> bool:
        return self.jersey_number in UNKNOWN_NUMBERS

    def with_updates(self, **updates: Any) -> "PlayerBox":
        data = asdict(self)
        data.update(updates)
        data["bbox_xywh"] = tuple(data["bbox_xywh"])
        return PlayerBox(**data)

    def to_manifest(self, crop_path: str | None = None) -> dict[str, Any]:
        row = asdict(self)
        row["bbox_xywh"] = list(self.bbox_xywh)
        row["team_side"] = self.team_side
        row["gallery_entity_id"] = self.gallery_entity_id or self.match_id
        row["stable_team_id"] = self.stable_team_id
        row["identity_id"] = self.identity_id
        row["is_unknown_number"] = self.is_unknown_number
        if crop_path is not None:
            row["crop_path"] = crop_path
        return row


def parse_video_id(stem: str) -> tuple[str, str]:
    match = PLAY_RE.match(stem)
    if not match:
        return stem, "play0"
    return match.group("match"), f"play{match.group('play')}"


def load_player_boxes(annotation_path: Path) -> list[PlayerBox]:
    data = json.loads(annotation_path.read_text())
    match_id, play_id = parse_video_id(annotation_path.stem)
    images = {image["id"]: image for image in data.get("images", [])}
    boxes: list[PlayerBox] = []
    for ann in data.get("annotations", []):
        category_id = int(ann.get("category_id", -1))
        if category_id not in (NEAR_SIDE_CATEGORY, FAR_SIDE_CATEGORY):
            continue
        image = images.get(ann.get("image_id"))
        bbox = ann.get("bbox", [])
        if not image or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        attrs = ann.get("attributes", {})
        frame_file = str(image["file_name"])
        boxes.append(
            PlayerBox(
                ann_id=int(ann["id"]),
                image_id=int(ann["image_id"]),
                frame_file=frame_file,
                frame_index=int(Path(frame_file).stem),
                frame_width=int(image["width"]),
                frame_height=int(image["height"]),
                category_id=category_id,
                bbox_xywh=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                area=float(ann.get("area", bbox[2] * bbox[3])),
                match_id=match_id,
                play_id=play_id,
                video_id=annotation_path.stem,
                track_id=int(attrs.get("track_id", -1)),
                jersey_number=str(attrs.get("jersey_number", "")),
                jersey_visible=bool(attrs.get("jersey_visible", False)),
                occluded=bool(attrs.get("occluded", False)),
                jump=bool(attrs.get("jump", False)),
                keyframe=bool(attrs.get("keyframe", False)),
            )
        )
    return boxes


def iou_xywh(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    inter_w = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    inter_h = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = lw * lh + rw * rh - inter
    return inter / union if union > 0 else 0.0


def target_in_front(target: PlayerBox, other: PlayerBox, y_tolerance: float = 3.0) -> bool:
    if target.category_id == NEAR_SIDE_CATEGORY and other.category_id == FAR_SIDE_CATEGORY:
        return True
    if target.category_id == FAR_SIDE_CATEGORY and other.category_id == NEAR_SIDE_CATEGORY:
        return False
    target_y2 = target.adjusted_y2 if target.adjusted_y2 is not None else target.raw_y2
    other_y2 = other.adjusted_y2 if other.adjusted_y2 is not None else other.raw_y2
    return target_y2 + y_tolerance >= other_y2


def apply_jump_adjusted_y2(boxes: Iterable[PlayerBox]) -> list[PlayerBox]:
    by_track: dict[tuple[str, int], list[PlayerBox]] = defaultdict(list)
    for box in boxes:
        by_track[(box.video_id, box.track_id)].append(box)

    adjusted_by_ann: dict[tuple[str, int], float] = {}
    for track_boxes in by_track.values():
        ordered = sorted(track_boxes, key=lambda box: box.frame_index)
        for idx, box in enumerate(ordered):
            key = (box.video_id, box.ann_id)
            if not box.jump:
                adjusted_by_ann[key] = box.raw_y2
                continue
            prev_box = next((ordered[i] for i in range(idx - 1, -1, -1) if not ordered[i].jump), None)
            next_box = next((ordered[i] for i in range(idx + 1, len(ordered)) if not ordered[i].jump), None)
            if not prev_box or not next_box or next_box.frame_index == prev_box.frame_index:
                adjusted_by_ann[key] = box.raw_y2
                continue
            ratio = (box.frame_index - prev_box.frame_index) / (next_box.frame_index - prev_box.frame_index)
            adjusted_by_ann[key] = prev_box.raw_y2 + ratio * (next_box.raw_y2 - prev_box.raw_y2)
        # If duplicate frame annotations somehow occur, preserve a raw-y2 fallback.
        for box in ordered:
            adjusted_by_ann.setdefault((box.video_id, box.ann_id), box.raw_y2)

    return [
        box.with_updates(adjusted_y2=adjusted_by_ann.get((box.video_id, box.ann_id), box.raw_y2))
        for box in boxes
    ]


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def assign_stable_team_ids(boxes: Iterable[PlayerBox]) -> list[PlayerBox]:
    by_match_video_side: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for box in boxes:
        if not box.is_unknown_number:
            by_match_video_side[(box.match_id, box.video_id, box.category_id)].add(box.jersey_number)

    assignments: dict[tuple[str, str, int], str] = {}
    by_match: dict[str, list[str]] = defaultdict(list)
    for match_id, video_id, _ in by_match_video_side:
        if video_id not in by_match[match_id]:
            by_match[match_id].append(video_id)

    for match_id, video_ids in by_match.items():
        ordered_video_ids = sorted(video_ids)
        if not ordered_video_ids:
            continue

        base_video_id = ordered_video_ids[0]
        team_rosters = {
            "A": set(by_match_video_side.get((match_id, base_video_id, NEAR_SIDE_CATEGORY), set())),
            "B": set(by_match_video_side.get((match_id, base_video_id, FAR_SIDE_CATEGORY), set())),
        }
        base_overlap = team_rosters["A"] & team_rosters["B"]
        base_is_ambiguous = len(base_overlap) >= 3 or jaccard(team_rosters["A"], team_rosters["B"]) >= 0.25
        if not base_is_ambiguous:
            assignments[(match_id, base_video_id, NEAR_SIDE_CATEGORY)] = "A"
            assignments[(match_id, base_video_id, FAR_SIDE_CATEGORY)] = "B"

        for video_id in ordered_video_ids[1:]:
            near_roster = by_match_video_side.get((match_id, video_id, NEAR_SIDE_CATEGORY), set())
            far_roster = by_match_video_side.get((match_id, video_id, FAR_SIDE_CATEGORY), set())
            if base_is_ambiguous:
                continue

            same = (jaccard(near_roster, team_rosters["A"]) + jaccard(far_roster, team_rosters["B"])) / 2.0
            swap = (jaccard(near_roster, team_rosters["B"]) + jaccard(far_roster, team_rosters["A"])) / 2.0
            if abs(same - swap) < 0.15:
                continue
            if swap > same:
                near_team, far_team = "B", "A"
            else:
                near_team, far_team = "A", "B"
            assignments[(match_id, video_id, NEAR_SIDE_CATEGORY)] = near_team
            assignments[(match_id, video_id, FAR_SIDE_CATEGORY)] = far_team
            team_rosters[near_team].update(near_roster)
            team_rosters[far_team].update(far_roster)

    return [
        box.with_updates(
            gallery_entity_id=box.match_id
            if (box.match_id, box.video_id, box.category_id) in assignments
            else box.video_id,
            stable_team_id=assignments.get((box.match_id, box.video_id, box.category_id), box.team_side),
        )
        for box in boxes
    ]


def apply_salience(boxes: Iterable[PlayerBox], iou_threshold: float = 0.40) -> list[PlayerBox]:
    by_frame: dict[tuple[str, int], list[PlayerBox]] = defaultdict(list)
    for box in boxes:
        by_frame[(box.video_id, box.image_id)].append(box)

    scored: list[PlayerBox] = []
    for frame_boxes in by_frame.values():
        for target in frame_boxes:
            overlaps = [
                (other, iou_xywh(target.bbox_xywh, other.bbox_xywh))
                for other in frame_boxes
                if other.ann_id != target.ann_id
            ]
            max_iou = max((score for _, score in overlaps), default=0.0)
            high = [(other, score) for other, score in overlaps if score >= iou_threshold]
            if not high:
                reason = "low_iou"
                front = True
            else:
                front = all(target_in_front(target, other) for other, _ in high)
                reason = "front_overlap" if front else "behind_overlap"
            scored.append(
                target.with_updates(
                    max_iou=max_iou,
                    high_iou_count=len(high),
                    front_of_high_iou=front,
                    salience_reason=reason,
                )
            )
    return scored


def apply_edge_filter(boxes: Iterable[PlayerBox], margin_px: float = 2.0) -> list[PlayerBox]:
    filtered: list[PlayerBox] = []
    for box in boxes:
        x, y, width, height = box.bbox_xywh
        touches_edge = (
            x <= margin_px
            or y <= margin_px
            or x + width >= box.frame_width - margin_px
            or y + height >= box.frame_height - margin_px
        )
        filtered.append(
            box.with_updates(salience_reason="edge_touch" if touches_edge else box.salience_reason)
        )
    return filtered


def xywh_to_xyxy(bbox_xywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, width, height = bbox_xywh
    return x, y, x + width, y + height


def upper_half_xyxy(bbox_xywh: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, width, height = bbox_xywh
    return x, y, x + width, y + height / 2.0


def area_xyxy(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_xyxy(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def union_area_xyxy(rects: list[tuple[float, float, float, float]]) -> float:
    if not rects:
        return 0.0
    xs = sorted({coord for rect in rects for coord in (rect[0], rect[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = sorted((rect[1], rect[3]) for rect in rects if rect[0] < right and rect[2] > left)
        if not intervals:
            continue
        covered = 0.0
        cur_top, cur_bottom = intervals[0]
        for top, bottom in intervals[1:]:
            if top > cur_bottom:
                covered += cur_bottom - cur_top
                cur_top, cur_bottom = top, bottom
            else:
                cur_bottom = max(cur_bottom, bottom)
        covered += cur_bottom - cur_top
        area += (right - left) * covered
    return area


def apply_crop_contamination(
    boxes: Iterable[PlayerBox],
    upper_frac_threshold: float = 0.60,
    target_coverage_threshold: float = 0.45,
    aggregate_overlap_threshold: float = 0.70,
) -> list[PlayerBox]:
    by_frame: dict[tuple[str, int], list[PlayerBox]] = defaultdict(list)
    for box in boxes:
        by_frame[(box.video_id, box.image_id)].append(box)

    scored: list[PlayerBox] = []
    for frame_boxes in by_frame.values():
        for target in frame_boxes:
            target_crop = xywh_to_xyxy(target.bbox_xywh)
            target_area = area_xyxy(target_crop)
            max_upper_frac = 0.0
            max_target_coverage = 0.0
            target_intersections: list[tuple[float, float, float, float]] = []
            sum_intersection_area = 0.0
            for other in frame_boxes:
                if other.ann_id == target.ann_id:
                    continue
                target_intersection = intersection_xyxy(target_crop, xywh_to_xyxy(other.bbox_xywh))
                if target_intersection is not None:
                    target_coverage = area_xyxy(target_intersection) / max(1.0, target_area)
                    max_target_coverage = max(max_target_coverage, target_coverage)
                    target_intersections.append(target_intersection)
                    sum_intersection_area += area_xyxy(target_intersection)
                upper = upper_half_xyxy(other.bbox_xywh)
                clipped = intersection_xyxy(target_crop, upper)
                if clipped is None:
                    continue
                upper_frac = area_xyxy(clipped) / max(1.0, area_xyxy(upper))
                max_upper_frac = max(max_upper_frac, upper_frac)
            union_frac = union_area_xyxy(target_intersections) / max(1.0, target_area)
            sum_frac = sum_intersection_area / max(1.0, target_area)
            reason = target.salience_reason
            if max_target_coverage >= target_coverage_threshold:
                reason = "target_coverage"
            elif union_frac >= aggregate_overlap_threshold:
                reason = "aggregate_overlap"
            elif max_upper_frac >= upper_frac_threshold:
                reason = "crop_contamination"
            scored.append(
                target.with_updates(
                    max_other_target_coverage=max_target_coverage,
                    other_bbox_union_frac=union_frac,
                    other_bbox_sum_frac=sum_frac,
                    max_other_upper_frac_in_crop=max_upper_frac,
                    salience_reason=reason,
                )
            )
    return scored


def is_acceptable_candidate(box: PlayerBox) -> bool:
    return box.salience_reason in {"low_iou", "front_overlap"}


def quality_score(box: PlayerBox) -> tuple[float, ...]:
    return (
        1.0 if not box.occluded else 0.0,
        1.0 if box.jersey_visible else 0.0,
        1.0 if box.salience_reason == "low_iou" else 0.0,
        -box.max_iou,
        math.log(max(1.0, box.area)),
        1.0 if box.keyframe else 0.0,
        -float(box.frame_index),
    )


def quantile_bucket(value: float, low: float, high: float, buckets: int) -> int:
    if high <= low:
        return 0
    ratio = (value - low) / (high - low)
    return max(0, min(buckets - 1, int(ratio * buckets)))


def pose_bucket(box: PlayerBox) -> str:
    _, _, width, height = box.bbox_xywh
    ratio = height / max(1.0, width)
    if ratio >= 2.8:
        return "tall"
    if ratio >= 2.0:
        return "mid"
    return "wide"


def diversity_key(box: PlayerBox, group: list[PlayerBox]) -> tuple[Any, ...]:
    frames = [item.frame_index for item in group]
    xs = [item.bbox_xywh[0] + item.bbox_xywh[2] / 2 for item in group]
    frame_bucket = quantile_bucket(box.frame_index, min(frames), max(frames), 5)
    x_bucket = quantile_bucket(box.bbox_xywh[0] + box.bbox_xywh[2] / 2, min(xs), max(xs), 3)
    return (box.play_id, frame_bucket, x_bucket, bool(box.jump), pose_bucket(box))


def _norm(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return (value - low) / (high - low)


def _visual_features(box: PlayerBox, group_stats: dict[str, Any]) -> tuple[float, ...]:
    x, y, width, height = box.bbox_xywh
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    aspect = height / max(1.0, width)
    play_idx = group_stats["play_index"][box.play_id]
    return (
        2.5 * _norm(play_idx, 0, max(1, group_stats["play_count"] - 1)),
        1.3 * _norm(box.frame_index, group_stats["frame_min"], group_stats["frame_max"]),
        1.5 * _norm(center_x, group_stats["x_min"], group_stats["x_max"]),
        0.9 * _norm(center_y, group_stats["y_min"], group_stats["y_max"]),
        0.8 * _norm(width, group_stats["w_min"], group_stats["w_max"]),
        0.8 * _norm(height, group_stats["h_min"], group_stats["h_max"]),
        1.0 * _norm(aspect, group_stats["aspect_min"], group_stats["aspect_max"]),
        0.8 if box.jump else 0.0,
    )


def _feature_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _group_stats(pool: list[PlayerBox]) -> dict[str, Any]:
    centers_x = [box.bbox_xywh[0] + box.bbox_xywh[2] / 2.0 for box in pool]
    centers_y = [box.bbox_xywh[1] + box.bbox_xywh[3] / 2.0 for box in pool]
    widths = [box.bbox_xywh[2] for box in pool]
    heights = [box.bbox_xywh[3] for box in pool]
    aspects = [box.bbox_xywh[3] / max(1.0, box.bbox_xywh[2]) for box in pool]
    plays = sorted({box.play_id for box in pool})
    return {
        "play_index": {play_id: idx for idx, play_id in enumerate(plays)},
        "play_count": len(plays),
        "frame_min": min(box.frame_index for box in pool),
        "frame_max": max(box.frame_index for box in pool),
        "x_min": min(centers_x),
        "x_max": max(centers_x),
        "y_min": min(centers_y),
        "y_max": max(centers_y),
        "w_min": min(widths),
        "w_max": max(widths),
        "h_min": min(heights),
        "h_max": max(heights),
        "aspect_min": min(aspects),
        "aspect_max": max(aspects),
    }


def _quality_ranks(pool: list[PlayerBox]) -> dict[int, float]:
    ordered = sorted(pool, key=quality_score, reverse=True)
    if len(ordered) == 1:
        return {ordered[0].ann_id: 1.0}
    return {
        box.ann_id: 1.0 - idx / (len(ordered) - 1)
        for idx, box in enumerate(ordered)
    }


def apply_area_floor(pool: list[PlayerBox], target_count: int) -> list[PlayerBox]:
    if len(pool) > target_count:
        areas = sorted(box.area for box in pool)
        median_area = areas[len(areas) // 2]
        larger_pool = [box for box in pool if box.area >= median_area]
        if len(larger_pool) >= target_count:
            return larger_pool
    return pool


def select_diverse_from_pool(pool: list[PlayerBox], target_count: int, rng: random.Random) -> list[PlayerBox]:
    if len(pool) <= target_count:
        return sorted(pool, key=lambda box: (box.play_id, box.frame_index, box.track_id))

    shuffled = list(pool)
    rng.shuffle(shuffled)
    shuffled.sort(key=quality_score, reverse=True)
    stats = _group_stats(shuffled)
    features = {box.ann_id: _visual_features(box, stats) for box in shuffled}
    quality_ranks = _quality_ranks(shuffled)
    selected = [shuffled[0]]
    seen = {shuffled[0].ann_id}

    while len(selected) < target_count:
        best: PlayerBox | None = None
        best_score = float("-inf")
        for candidate in shuffled:
            if candidate.ann_id in seen:
                continue
            min_dist = min(
                _feature_distance(features[candidate.ann_id], features[pick.ann_id])
                for pick in selected
            )
            score = min_dist + 0.15 * quality_ranks[candidate.ann_id]
            if score > best_score:
                best = candidate
                best_score = score
        if best is None:
            break
        selected.append(best)
        seen.add(best.ann_id)

    return selected[:target_count]


def play_quotas(pool: list[PlayerBox], target_count: int) -> dict[str, int]:
    by_play: dict[str, list[PlayerBox]] = defaultdict(list)
    for box in pool:
        by_play[box.play_id].append(box)
    plays = sorted(by_play)
    if len(plays) <= 1:
        return {plays[0]: min(target_count, len(by_play[plays[0]]))} if plays else {}

    quotas = {play_id: min(len(by_play[play_id]), target_count // len(plays)) for play_id in plays}
    remaining = target_count - sum(quotas.values())
    while remaining > 0:
        changed = False
        for play_id in sorted(plays, key=lambda item: (-len(by_play[item]), item)):
            if quotas[play_id] >= len(by_play[play_id]):
                continue
            quotas[play_id] += 1
            remaining -= 1
            changed = True
            if remaining == 0:
                break
        if not changed:
            break
    return quotas


def select_diverse_candidates(group: list[PlayerBox], target_count: int, rng: random.Random) -> list[PlayerBox]:
    acceptable = [box for box in group if is_acceptable_candidate(box)]
    pool = apply_area_floor(acceptable or group, target_count)
    if len(pool) <= target_count:
        return sorted(pool, key=lambda box: (box.play_id, box.frame_index, box.track_id))

    quotas = play_quotas(pool, target_count)
    if len(quotas) <= 1:
        return select_diverse_from_pool(pool, target_count, rng)

    by_play: dict[str, list[PlayerBox]] = defaultdict(list)
    for box in pool:
        by_play[box.play_id].append(box)

    selected: list[PlayerBox] = []
    seen: set[int] = set()
    for play_id in sorted(quotas):
        quota = quotas[play_id]
        if quota <= 0:
            continue
        picks = select_diverse_from_pool(by_play[play_id], quota, rng)
        selected.extend(picks)
        seen.update(box.ann_id for box in picks)

    if len(selected) < min(target_count, len(pool)):
        remainder = [box for box in pool if box.ann_id not in seen]
        selected.extend(select_diverse_from_pool(remainder, target_count - len(selected), rng))

    return selected[:target_count]


def expanded_bbox(
    bbox_xywh: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    expand: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox_xywh
    pad_x = width * expand
    pad_y = height * expand
    x1 = max(0, int(round(x - pad_x)))
    y1 = max(0, int(round(y - pad_y)))
    x2 = min(frame_width, int(round(x + width + pad_x)))
    y2 = min(frame_height, int(round(y + height + pad_y)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def crop_image(box: PlayerBox, frames_dir: Path, expand: float) -> Image.Image:
    frame_path = frames_dir / box.video_id / box.frame_file
    with Image.open(frame_path) as image:
        x1, y1, x2, y2 = expanded_bbox(box.bbox_xywh, box.frame_width, box.frame_height, expand)
        return image.convert("RGB").crop((x1, y1, x2, y2))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def crop_rel_path(box: PlayerBox) -> Path:
    name = (
        f"{box.video_id}__{box.team_side}__jersey{safe_name(box.jersey_number or 'blank')}"
        f"__frame{box.frame_index:06d}__ann{box.ann_id:07d}.png"
    )
    return Path(box.match_id) / box.identity_id / name


def materialize_crops(
    selections: dict[str, list[PlayerBox]],
    frames_dir: Path,
    crops_dir: Path,
    expand: float,
    jpeg_quality: int,
    num_workers: int = 1,
) -> list[dict[str, Any]]:
    tasks = []
    for identity_id, boxes in sorted(selections.items()):
        for rank, box in enumerate(boxes):
            rel_path = crop_rel_path(box)
            tasks.append((identity_id, rank, box, rel_path, frames_dir, crops_dir, expand))
    if num_workers <= 1 or len(tasks) <= 1:
        return [_materialize_one_crop(task) for task in tasks]
    with Pool(processes=num_workers) as pool:
        return list(pool.map(_materialize_one_crop, tasks, chunksize=32))


def valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _materialize_one_crop(task: tuple[str, int, PlayerBox, Path, Path, Path, float]) -> dict[str, Any]:
    identity_id, rank, box, rel_path, frames_dir, crops_dir, expand = task
    out_path = crops_dir / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not valid_image(out_path):
        tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
        crop = crop_image(box, frames_dir, expand)
        crop.save(tmp_path, format="PNG", optimize=True)
        tmp_path.replace(out_path)
    row = box.to_manifest(str(rel_path))
    row["sample_rank"] = rank
    row["selected_for_identity"] = identity_id
    return row


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        font_path = Path(path)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def fit_thumb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail(size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", size, (242, 242, 242))
    x = (size[0] - thumb.width) // 2
    y = (size[1] - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def write_match_grid(
    match_id: str,
    selections: dict[str, list[PlayerBox]],
    frames_dir: Path,
    out_path: Path,
    target_count: int,
    expand: float,
) -> None:
    row_label_w = 190
    thumb_w, thumb_h = 180, 260
    cell_pad = 6
    label_h = 34
    header_h = 38
    row_h = thumb_h + label_h + cell_pad * 2
    grid_w = row_label_w + target_count * thumb_w
    identities = sorted(selections)
    grid_h = header_h + max(1, len(identities)) * row_h
    sheet = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(12)
    small = load_font(10)
    title_font = load_font(16)
    draw.rectangle((0, 0, grid_w, header_h), fill=(32, 36, 40))
    draw.text((10, 9), f"{match_id} gallery", fill="white", font=title_font)
    for col in range(target_count):
        x = row_label_w + col * thumb_w
        draw.text((x + 4, 12), str(col + 1), fill=(220, 220, 220), font=small)

    for row_idx, identity_id in enumerate(identities):
        y = header_h + row_idx * row_h
        fill = (248, 249, 250) if row_idx % 2 == 0 else (238, 241, 244)
        draw.rectangle((0, y, grid_w, y + row_h), fill=fill)
        parts = identity_id.split("__")
        row_label = f"{parts[-2]} {parts[-1].replace('jersey', '#')}"
        draw.text((10, y + 12), row_label, fill=(20, 20, 20), font=font)
        draw.text((10, y + 32), f"{len(selections[identity_id])}/{target_count}", fill=(80, 80, 80), font=small)
        for col, box in enumerate(selections[identity_id]):
            x = row_label_w + col * thumb_w
            try:
                crop = crop_image(box, frames_dir, expand)
                thumb = fit_thumb(crop, (thumb_w - cell_pad * 2, thumb_h))
                sheet.paste(thumb, (x + cell_pad, y + cell_pad))
            except FileNotFoundError:
                draw.rectangle((x + cell_pad, y + cell_pad, x + thumb_w - cell_pad, y + thumb_h), outline="red")
            side = "N" if box.team_side == "near" else "F"
            label = f"P{box.play_id[-1]} F{box.frame_index} {side}"
            diag = f"i{box.max_iou:.2f} {'j' if box.jump else '-'} {'v' if box.jersey_visible else '-'}"
            label_y = y + thumb_h + cell_pad
            draw.rectangle((x + cell_pad, label_y, x + thumb_w - cell_pad, label_y + 15), fill=(32, 36, 40))
            draw.text((x + cell_pad + 3, label_y + 2), label, fill="white", font=small)
            draw.text((x + cell_pad + 3, label_y + 18), diag, fill=(70, 70, 70), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG", optimize=True)


def write_unknown_grid(
    unknowns: list[PlayerBox],
    frames_dir: Path,
    out_path: Path,
    expand: float,
    max_items: int = 120,
) -> None:
    boxes = sorted(unknowns, key=lambda box: (box.match_id, box.play_id, box.team_side, box.frame_index))[:max_items]
    if not boxes:
        return
    cols = 10
    thumb_w, thumb_h = 180, 240
    label_h = 42
    header_h = 34
    rows = (len(boxes) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, header_h + rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(10)
    title_font = load_font(15)
    draw.rectangle((0, 0, sheet.width, header_h), fill=(32, 36, 40))
    draw.text((10, 8), f"Unknown-number QA ({len(boxes)} shown)", fill="white", font=title_font)
    for idx, box in enumerate(boxes):
        x = (idx % cols) * thumb_w
        y = header_h + (idx // cols) * (thumb_h + label_h)
        crop = crop_image(box, frames_dir, expand)
        sheet.paste(fit_thumb(crop, (thumb_w - 8, thumb_h)), (x + 4, y + 4))
        label = f"{box.match_id[:6]} {box.play_id} {box.team_side}"
        diag = f"num={box.jersey_number or 'blank'} f{box.frame_index} i{box.max_iou:.2f}"
        draw.text((x + 4, y + thumb_h + 6), label, fill=(20, 20, 20), font=font)
        draw.text((x + 4, y + thumb_h + 22), diag, fill=(70, 70, 70), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG", optimize=True)


def first_train_matches(data_root: Path, count: int) -> list[str]:
    split_path = data_root / "splits" / "v0.json"
    split = json.loads(split_path.read_text())
    matches: list[str] = []
    for video_id in split["train"]:
        match_id, _ = parse_video_id(video_id)
        if match_id not in matches:
            matches.append(match_id)
        if len(matches) >= count:
            break
    return matches


def first_train_videos(data_root: Path, count: int) -> list[str]:
    split_path = data_root / "splits" / "v0.json"
    split = json.loads(split_path.read_text())
    return list(split["train"][:count])


def build_trial(
    data_root: Path,
    output_dir: Path,
    video_ids: list[str],
    group_scope: str,
    target_count: int,
    seed: int,
    iou_threshold: float,
    bbox_expand: float,
    jpeg_quality: int,
    num_workers: int,
) -> dict[str, Any]:
    ann_dir = data_root / "annotations" / "v0"
    frames_dir = data_root / "images_full"
    all_boxes: list[PlayerBox] = []
    for video_id in video_ids:
        all_boxes.extend(load_player_boxes(ann_dir / f"{video_id}.json"))

    if group_scope == "match":
        grouped_boxes = assign_stable_team_ids(all_boxes)
        entity_ids = sorted({box.gallery_entity_id or box.match_id for box in grouped_boxes})
    elif group_scope == "video":
        grouped_boxes = [
            box.with_updates(gallery_entity_id=box.video_id, stable_team_id=box.team_side)
            for box in all_boxes
        ]
        entity_ids = list(video_ids)
    else:
        raise ValueError(f"unsupported group scope: {group_scope}")

    adjusted = apply_jump_adjusted_y2(grouped_boxes)
    scored = apply_edge_filter(apply_crop_contamination(apply_salience(adjusted, iou_threshold=iou_threshold)))

    known_by_identity: dict[str, list[PlayerBox]] = defaultdict(list)
    unknowns: list[PlayerBox] = []
    for box in scored:
        if box.is_unknown_number:
            unknowns.append(box)
        else:
            known_by_identity[box.identity_id].append(box)

    rng = random.Random(seed)
    selections = {
        identity_id: select_diverse_candidates(group, target_count, rng)
        for identity_id, group in sorted(known_by_identity.items())
    }

    crop_rows = materialize_crops(
        selections,
        frames_dir,
        output_dir / "crops",
        bbox_expand,
        jpeg_quality,
        num_workers=num_workers,
    )
    write_jsonl(output_dir / "manifests" / "gallery_samples.jsonl", crop_rows)
    write_jsonl(output_dir / "manifests" / "unknown_number_candidates.jsonl", [box.to_manifest() for box in unknowns])

    for entity_id in entity_ids:
        entity_selections = {
            identity_id: boxes
            for identity_id, boxes in selections.items()
            if identity_id.startswith(f"{entity_id}__")
        }
        write_match_grid(
            match_id=entity_id,
            selections=entity_selections,
            frames_dir=frames_dir,
            out_path=output_dir / "reports" / f"{entity_id}_gallery.png",
            target_count=target_count,
            expand=bbox_expand,
        )
    write_unknown_grid(unknowns, frames_dir, output_dir / "reports" / "unknown_number_qa.png", bbox_expand)

    rejection_reasons = Counter(box.salience_reason for box in scored if not is_acceptable_candidate(box))
    summary = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "group_scope": group_scope,
        "videos": video_ids,
        "matches": sorted({parse_video_id(video_id)[0] for video_id in video_ids}),
        "gallery_entities": entity_ids,
        "target_count": target_count,
        "seed": seed,
        "iou_threshold": iou_threshold,
        "bbox_expand": bbox_expand,
        "num_workers": num_workers,
        "boxes_loaded": len(all_boxes),
        "known_identities": len(known_by_identity),
        "unknown_number_boxes": len(unknowns),
        "selected_crops": len(crop_rows),
        "short_identity_count": sum(1 for boxes in selections.values() if len(boxes) < target_count),
        "rejection_reasons": dict(rejection_reasons),
        "identity_summary": {
            identity_id: {
                "candidate_count": len(known_by_identity[identity_id]),
                "acceptable_count": sum(1 for box in known_by_identity[identity_id] if is_acceptable_candidate(box)),
                "selected_count": len(selections[identity_id]),
            }
            for identity_id in sorted(known_by_identity)
        },
    }
    write_json(output_dir / "reports" / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build volleyball player galleries.")
    parser.add_argument("--data-root", default="/mnt/t/data/vball/skyball/unified_dataset")
    parser.add_argument("--output", default="/mnt/t/output/jersey_gallery/three_play_trial_001")
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--video-count", type=int, default=3)
    parser.add_argument("--group-scope", choices=["video", "match"], default="video")
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--match-count", type=int, default=2)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--split-names", nargs="*", default=None)
    parser.add_argument("--target-count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--iou-threshold", type=float, default=0.40)
    parser.add_argument("--bbox-expand", type=float, default=0.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel crop materialization workers. Existing valid crops are reused.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.split_file:
        split_path = Path(args.split_file)
        split = json.loads(split_path.read_text())
        split_names = args.split_names or [name for name in split if name != "annotations"]
        summaries = {}
        for split_name in split_names:
            video_ids = list(split[split_name])
            summaries[split_name] = build_trial(
                data_root=data_root,
                output_dir=Path(args.output) / split_name,
                video_ids=video_ids,
                group_scope=args.group_scope,
                target_count=args.target_count,
                seed=args.seed,
                iou_threshold=args.iou_threshold,
                bbox_expand=args.bbox_expand,
                jpeg_quality=args.jpeg_quality,
                num_workers=args.num_workers,
            )
        write_json(Path(args.output) / "summary.json", summaries)
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return

    if args.videos:
        video_ids = args.videos
    elif args.matches:
        ann_dir = data_root / "annotations" / "v0"
        video_ids = [
            path.stem
            for match_id in args.matches
            for path in sorted(ann_dir.glob(f"{match_id}_play*.json"))
        ]
    elif args.group_scope == "match":
        ann_dir = data_root / "annotations" / "v0"
        video_ids = [
            path.stem
            for match_id in first_train_matches(data_root, args.match_count)
            for path in sorted(ann_dir.glob(f"{match_id}_play*.json"))
        ]
    else:
        video_ids = first_train_videos(data_root, args.video_count)
    summary = build_trial(
        data_root=data_root,
        output_dir=Path(args.output),
        video_ids=video_ids,
        group_scope=args.group_scope,
        target_count=args.target_count,
        seed=args.seed,
        iou_threshold=args.iou_threshold,
        bbox_expand=args.bbox_expand,
        jpeg_quality=args.jpeg_quality,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
