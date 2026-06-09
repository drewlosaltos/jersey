from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from .data import PlayerAnn


@dataclass(frozen=True)
class SampledAnn:
    kind: str
    rank: int
    ann: PlayerAnn


def _quality_score(ann: PlayerAnn) -> tuple[int, int, float]:
    visible = 1 if ann.jersey_visible_attr else 0
    clear = 0 if ann.occluded else 1
    return clear, visible, ann.area


def _temporally_diverse(records: list[PlayerAnn], count: int) -> list[PlayerAnn]:
    if len(records) <= count:
        return records
    sorted_records = sorted(records, key=lambda a: a.frame_index)
    if count == 1:
        return [max(sorted_records, key=_quality_score)]
    buckets: list[list[PlayerAnn]] = [[] for _ in range(count)]
    first = sorted_records[0].frame_index
    last = sorted_records[-1].frame_index
    span = max(1, last - first + 1)
    for rec in sorted_records:
        idx = min(count - 1, int((rec.frame_index - first) / span * count))
        buckets[idx].append(rec)
    picked: list[PlayerAnn] = []
    used: set[int] = set()
    for bucket in buckets:
        if not bucket:
            continue
        best = max(bucket, key=_quality_score)
        picked.append(best)
        used.add(best.ann_id)
    if len(picked) < count:
        leftovers = [r for r in sorted(records, key=_quality_score, reverse=True) if r.ann_id not in used]
        picked.extend(leftovers[: count - len(picked)])
    return sorted(picked[:count], key=lambda a: a.frame_index)


def _jersey_stratified(records: list[PlayerAnn], count: int, rng: random.Random) -> list[PlayerAnn]:
    visible = [r for r in records if r.jersey_visible_attr]
    non_visible = [r for r in records if not r.jersey_visible_attr]
    large = sorted(records, key=lambda r: r.area, reverse=True)[: max(count * 2, count)]
    pools = [visible, large, non_visible]
    picks: list[PlayerAnn] = []
    seen: set[int] = set()
    per_pool = max(1, count // len(pools))
    for pool in pools:
        if not pool:
            continue
        pool = sorted(pool, key=lambda r: (r.frame_index, -r.area))
        candidates = _temporally_diverse(pool, per_pool)
        for rec in candidates:
            if rec.ann_id not in seen:
                picks.append(rec)
                seen.add(rec.ann_id)
    if len(picks) < count:
        remaining = [r for r in records if r.ann_id not in seen]
        rng.shuffle(remaining)
        remaining.sort(key=_quality_score, reverse=True)
        picks.extend(remaining[: count - len(picks)])
    return sorted(picks[:count], key=lambda a: (a.video_id, a.frame_index, a.track_id))


def sample_trial(
    records: list[PlayerAnn],
    reid_per_player: int,
    jersey_per_player: int,
    seed: int,
) -> tuple[list[SampledAnn], list[SampledAnn]]:
    rng = random.Random(seed)
    by_identity: dict[str, list[PlayerAnn]] = defaultdict(list)
    for record in records:
        if record.track_id < 0:
            continue
        by_identity[record.identity_id].append(record)

    reid_samples: list[SampledAnn] = []
    jersey_samples: list[SampledAnn] = []
    for identity_id in sorted(by_identity):
        group = by_identity[identity_id]
        reid_candidates = [r for r in group if not r.occluded] or group
        reid_picks = _temporally_diverse(sorted(reid_candidates, key=_quality_score, reverse=True), reid_per_player)
        jersey_picks = _jersey_stratified(group, jersey_per_player, rng)
        reid_samples.extend(SampledAnn("reid", i, ann) for i, ann in enumerate(reid_picks))
        jersey_samples.extend(SampledAnn("jersey", i, ann) for i, ann in enumerate(jersey_picks))
    return reid_samples, jersey_samples
