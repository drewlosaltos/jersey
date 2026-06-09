from __future__ import annotations

from pathlib import Path

from jersey_sdg.crops import expanded_bbox_xyxy
from jersey_sdg.data import load_player_annotations, parse_video_id
from jersey_sdg.sampling import sample_trial
from jersey_sdg.vlm import parse_vlm_json


ANN = Path("/mnt/t/data/vball/skyball/annotations/v0/000000_cosmic_dodo_play1.json")


def test_parse_video_id() -> None:
    assert parse_video_id("000000_cosmic_dodo_play1") == ("000000_cosmic_dodo", "play1")


def test_load_player_annotations_real_json() -> None:
    records = load_player_annotations(ANN)
    assert len(records) == 4656
    first = records[0]
    assert first.video_id == "000000_cosmic_dodo_play1"
    assert first.match_id == "000000_cosmic_dodo"
    assert first.frame_file == "000001.jpg"
    assert first.track_id == 0
    assert first.jersey_number == "34"


def test_expanded_bbox_clamps() -> None:
    assert expanded_bbox_xyxy((0, 0, 10, 20), 100, 100, 0.5) == (0, 0, 15, 30)
    assert expanded_bbox_xyxy((90, 90, 20, 20), 100, 100, 0.5) == (80, 80, 100, 100)


def test_sampler_is_deterministic() -> None:
    records = load_player_annotations(ANN)
    reid_a, jersey_a = sample_trial(records, 2, 3, 7)
    reid_b, jersey_b = sample_trial(records, 2, 3, 7)
    assert [s.ann.ann_id for s in reid_a] == [s.ann.ann_id for s in reid_b]
    assert [s.ann.ann_id for s in jersey_a] == [s.ann.ann_id for s in jersey_b]


def test_parse_vlm_json_variants() -> None:
    assert parse_vlm_json('{"visible": true, "number": 12, "confidence": 0.8, "reason": "front"}') == {
        "visible": True,
        "number": "12",
        "confidence": 0.8,
        "reason": "front",
    }
    parsed = parse_vlm_json('```json\n{"visible": false, "number": null, "confidence": 2}\n```')
    assert parsed["visible"] is False
    assert parsed["number"] is None
    assert parsed["confidence"] == 1.0
