from __future__ import annotations

from jersey_gallery import (
    FAR_SIDE_CATEGORY,
    NEAR_SIDE_CATEGORY,
    PlayerBox,
    apply_crop_contamination,
    apply_jump_adjusted_y2,
    assign_stable_team_ids,
    iou_xywh,
    parse_video_id,
    select_diverse_candidates,
    target_in_front,
)


def box(
    ann_id: int,
    *,
    category_id: int = NEAR_SIDE_CATEGORY,
    y: float = 10,
    height: float = 100,
    frame_index: int = 1,
    jump: bool = False,
    jersey_number: str = "7",
) -> PlayerBox:
    return PlayerBox(
        ann_id=ann_id,
        image_id=frame_index,
        frame_file=f"{frame_index:06d}.jpg",
        frame_index=frame_index,
        frame_width=1920,
        frame_height=1080,
        category_id=category_id,
        bbox_xywh=(100.0 + ann_id, y, 40.0, height),
        area=40.0 * height,
        match_id="m",
        play_id="play1",
        video_id="m_play1",
        track_id=0,
        jersey_number=jersey_number,
        jersey_visible=True,
        occluded=False,
        jump=jump,
        keyframe=True,
        adjusted_y2=y + height,
        salience_reason="low_iou",
    )


def test_parse_video_id() -> None:
    assert parse_video_id("000000_match_play3") == ("000000_match", "play3")


def test_iou_xywh() -> None:
    assert iou_xywh((0, 0, 10, 10), (20, 20, 5, 5)) == 0
    assert round(iou_xywh((0, 0, 10, 10), (5, 0, 10, 10)), 3) == 0.333


def test_target_in_front_near_over_far() -> None:
    near = box(1, category_id=NEAR_SIDE_CATEGORY, y=100)
    far = box(2, category_id=FAR_SIDE_CATEGORY, y=500)
    assert target_in_front(near, far)
    assert not target_in_front(far, near)


def test_target_in_front_same_side_y2() -> None:
    front = box(1, y=200, height=100)
    back = box(2, y=100, height=100)
    assert target_in_front(front, back)
    assert not target_in_front(back, front)


def test_jump_adjusted_y2_interpolates_between_non_jump_frames() -> None:
    records = [
        box(1, y=100, height=100, frame_index=1, jump=False),
        box(2, y=20, height=100, frame_index=2, jump=True),
        box(3, y=120, height=100, frame_index=3, jump=False),
    ]
    adjusted = {item.ann_id: item.adjusted_y2 for item in apply_jump_adjusted_y2(records)}
    assert adjusted[1] == 200
    assert adjusted[2] == 210
    assert adjusted[3] == 220


def test_jump_adjusted_y2_keys_annotation_ids_by_video() -> None:
    first = box(1, y=100, height=100, frame_index=1)
    second = box(1, y=300, height=100, frame_index=1).with_updates(video_id="other_play1")

    adjusted = apply_jump_adjusted_y2([first, second])

    by_video = {item.video_id: item.adjusted_y2 for item in adjusted}
    assert by_video["m_play1"] == 200
    assert by_video["other_play1"] == 400


def test_crop_contamination_rejects_other_upper_torso_inside_target_crop() -> None:
    target = box(1).with_updates(
        image_id=1,
        bbox_xywh=(100.0, 100.0, 100.0, 200.0),
        salience_reason="low_iou",
    )
    contaminant = box(2).with_updates(
        image_id=1,
        bbox_xywh=(130.0, 120.0, 80.0, 220.0),
        salience_reason="low_iou",
    )

    scored = apply_crop_contamination(
        [target, contaminant],
        upper_frac_threshold=0.60,
        target_coverage_threshold=1.1,
    )
    by_ann = {item.ann_id: item for item in scored}

    assert by_ann[1].salience_reason == "crop_contamination"
    assert by_ann[1].max_other_upper_frac_in_crop >= 0.60


def test_crop_contamination_rejects_aggregate_overlap() -> None:
    target = box(1).with_updates(
        image_id=1,
        bbox_xywh=(100.0, 100.0, 100.0, 200.0),
        salience_reason="low_iou",
    )
    left_overlap = box(2).with_updates(
        image_id=1,
        bbox_xywh=(80.0, 120.0, 80.0, 190.0),
        salience_reason="low_iou",
    )
    right_overlap = box(3).with_updates(
        image_id=1,
        bbox_xywh=(140.0, 120.0, 80.0, 190.0),
        salience_reason="low_iou",
    )

    scored = apply_crop_contamination(
        [target, left_overlap, right_overlap],
        target_coverage_threshold=1.1,
    )
    by_ann = {item.ann_id: item for item in scored}

    assert by_ann[1].salience_reason == "aggregate_overlap"
    assert by_ann[1].other_bbox_union_frac >= 0.70


def test_crop_contamination_rejects_single_box_covering_target() -> None:
    target = box(1).with_updates(
        image_id=1,
        bbox_xywh=(100.0, 100.0, 40.0, 80.0),
        salience_reason="low_iou",
    )
    large_other = box(2).with_updates(
        image_id=1,
        bbox_xywh=(80.0, 80.0, 120.0, 180.0),
        salience_reason="low_iou",
    )

    scored = apply_crop_contamination([target, large_other])
    by_ann = {item.ann_id: item for item in scored}

    assert by_ann[1].salience_reason == "target_coverage"
    assert by_ann[1].max_other_target_coverage == 1.0


def test_crop_contamination_rejects_mid_target_coverage() -> None:
    target = box(1).with_updates(
        image_id=1,
        bbox_xywh=(100.0, 100.0, 100.0, 100.0),
        salience_reason="low_iou",
    )
    mid_cover = box(2).with_updates(
        image_id=1,
        bbox_xywh=(100.0, 100.0, 46.0, 100.0),
        salience_reason="low_iou",
    )

    scored = apply_crop_contamination([target, mid_cover])
    by_ann = {item.ann_id: item for item in scored}

    assert by_ann[1].salience_reason == "target_coverage"
    assert by_ann[1].max_other_target_coverage > 0.45


def test_edge_filter_rejects_screen_edge_boxes() -> None:
    from jersey_gallery import apply_edge_filter

    edge = box(1).with_updates(bbox_xywh=(0.0, 100.0, 50.0, 120.0))
    interior = box(2).with_updates(bbox_xywh=(10.0, 100.0, 50.0, 120.0))

    scored = apply_edge_filter([edge, interior])
    by_ann = {item.ann_id: item for item in scored}

    assert by_ann[1].salience_reason == "edge_touch"
    assert by_ann[2].salience_reason == "low_iou"


def test_stable_team_assignment_handles_side_switches() -> None:
    records = [
        box(1, category_id=NEAR_SIDE_CATEGORY, jersey_number="1").with_updates(video_id="m_play1"),
        box(2, category_id=NEAR_SIDE_CATEGORY, jersey_number="2").with_updates(video_id="m_play1"),
        box(3, category_id=FAR_SIDE_CATEGORY, jersey_number="8").with_updates(video_id="m_play1"),
        box(4, category_id=FAR_SIDE_CATEGORY, jersey_number="9").with_updates(video_id="m_play1"),
        box(5, category_id=NEAR_SIDE_CATEGORY, jersey_number="8").with_updates(video_id="m_play2"),
        box(6, category_id=NEAR_SIDE_CATEGORY, jersey_number="9").with_updates(video_id="m_play2"),
        box(7, category_id=FAR_SIDE_CATEGORY, jersey_number="1").with_updates(video_id="m_play2"),
        box(8, category_id=FAR_SIDE_CATEGORY, jersey_number="2").with_updates(video_id="m_play2"),
    ]

    assigned = assign_stable_team_ids(records)
    by_ann = {item.ann_id: item for item in assigned}

    assert by_ann[1].stable_team_id == by_ann[7].stable_team_id
    assert by_ann[3].stable_team_id == by_ann[5].stable_team_id
    assert by_ann[1].stable_team_id != by_ann[3].stable_team_id
    assert by_ann[1].stable_team_id == "A"
    assert by_ann[3].stable_team_id == "B"
    assert by_ann[5].gallery_entity_id == "m"


def test_stable_team_assignment_keeps_high_overlap_match_play_local() -> None:
    records = [
        box(1, category_id=NEAR_SIDE_CATEGORY, jersey_number="1").with_updates(video_id="m_play1"),
        box(2, category_id=NEAR_SIDE_CATEGORY, jersey_number="2").with_updates(video_id="m_play1"),
        box(3, category_id=NEAR_SIDE_CATEGORY, jersey_number="3").with_updates(video_id="m_play1"),
        box(4, category_id=FAR_SIDE_CATEGORY, jersey_number="1").with_updates(video_id="m_play1"),
        box(5, category_id=FAR_SIDE_CATEGORY, jersey_number="2").with_updates(video_id="m_play1"),
        box(6, category_id=FAR_SIDE_CATEGORY, jersey_number="3").with_updates(video_id="m_play1"),
        box(7, category_id=NEAR_SIDE_CATEGORY, jersey_number="8").with_updates(video_id="m_play2"),
        box(8, category_id=FAR_SIDE_CATEGORY, jersey_number="9").with_updates(video_id="m_play2"),
    ]

    assigned = assign_stable_team_ids(records)
    by_ann = {item.ann_id: item for item in assigned}

    assert by_ann[1].gallery_entity_id == "m_play1"
    assert by_ann[1].stable_team_id == "near"
    assert by_ann[4].stable_team_id == "far"
    assert by_ann[7].gallery_entity_id == "m_play2"


def test_video_scoped_identity_uses_video_and_side() -> None:
    record = box(1, category_id=FAR_SIDE_CATEGORY, jersey_number="8").with_updates(
        video_id="m_play2",
        gallery_entity_id="m_play2",
        stable_team_id="far",
    )

    assert record.identity_id == "m_play2__far__jersey8"


def test_sampling_is_deterministic() -> None:
    import random

    group = [
        box(i, frame_index=i, y=10 + i, jersey_number="7")
        for i in range(1, 40)
    ]
    first = select_diverse_candidates(group, 20, random.Random(123))
    second = select_diverse_candidates(group, 20, random.Random(123))
    assert [item.ann_id for item in first] == [item.ann_id for item in second]
    assert len(first) == 20


def test_sampling_spreads_single_play_track_over_time() -> None:
    import random

    group = [
        box(i, frame_index=i, y=10 + (i % 5), jersey_number="7")
        for i in range(1, 101)
    ]

    selected = select_diverse_candidates(group, 10, random.Random(123))
    frames = [item.frame_index for item in selected]

    assert min(frames) <= 10
    assert max(frames) >= 90
    assert len({frame // 10 for frame in frames}) >= 7


def test_sampling_balances_available_match_plays() -> None:
    import random
    from collections import Counter

    group = []
    ann_id = 1
    for play_id in ("play1", "play2", "play3"):
        for frame_index in range(1, 31):
            group.append(
                box(ann_id, frame_index=frame_index, y=10 + frame_index, jersey_number="7").with_updates(
                    play_id=play_id,
                    video_id=f"m_{play_id}",
                )
            )
            ann_id += 1

    selected = select_diverse_candidates(group, 15, random.Random(123))
    by_play = Counter(item.play_id for item in selected)

    assert by_play == {"play1": 5, "play2": 5, "play3": 5}
