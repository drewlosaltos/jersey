# Jersey Gallery

Build player crop galleries from the SkyBall annotations for visual QA and re-id dataset construction.

## Strategy

- Treat each `match_play` video as the default gallery entity.
- Crop exactly to the ground-truth bbox; do not expand crops.
- Keep 15 crops per player identity.
- Reject bad crops with:
  - max target coverage by one other player >= `0.45`
  - aggregate target coverage by all other players >= `0.70`
  - another player's upper torso mostly inside the crop >= `0.60`
  - bbox touching the image edge
  - high IOU occlusion where the target is behind the occluder
- For same-side depth, larger adjusted `y2` is treated as in front.
- For jumping players, `y2` is linearly interpolated from nearby non-jump frames.
- Select final crops by farthest-point sampling over frame, position, bbox shape, and pose features.

For match-level team IDs, use `play1 near = team A` and `play1 far = team B`, then compare near/far jersey rosters in later plays. Only merge plays when roster assignment has a clear margin; otherwise keep labels play-local. Matches with high A/B jersey-number overlap should stay play-local unless another signal, such as jersey appearance, confirms the assignment.

## Data

Default data root:

```bash
/mnt/t/data/vball/skyball/unified_dataset
```

Expected annotations:

```bash
/mnt/t/data/vball/skyball/unified_dataset/annotations/v0/*_play*.json
```

Generated dataset galleries are written under `/mnt/t/data/vball/skyball/jersey/gallery/v0`.

## Run

Run tests:

```bash
PYTHONPATH=. pytest -q
```

Build a small three-match-play QA set:

```bash
python jersey_gallery.py \
  --output /mnt/t/output/jersey_gallery/three_match_play_trial_001 \
  --videos \
    000000_cosmic_dodo_play1 \
    000003_haughty_salamander_play1 \
    000004_intrepid_iguana_play1
```

Build ten more match-play grids:

```bash
python jersey_gallery.py \
  --output /mnt/t/output/jersey_gallery/ten_more_match_play_trial_001 \
  --videos \
    000005_boisterous_quetzal_play1 \
    000006_yappy_chevrotain_play1 \
    000007_goofy_aardvark_play1 \
    000008_blistering_hamster_play1 \
    000009_haunted_warthog_play1 \
    000010_turbulent_walrus_play1 \
    000011_fierce_piranha_play1 \
    000012_resplendent_viscacha_play1 \
    000013_gigantic_blobfish_play1 \
    000014_morose_okapia_play1
```

Outputs:

- `reports/*_gallery.png`: visual grid per match-play
- `manifests/gallery_samples.jsonl`: selected crop metadata
- `reports/summary.json`: counts and rejection reasons

Build the split-organized gallery dataset:

```bash
python jersey_gallery.py \
  --group-scope match \
  --data-root /mnt/t/data/vball/skyball/unified_dataset \
  --split-file /mnt/t/data/vball/skyball/unified_dataset/splits/v0.json \
  --split-names train val \
  --output /mnt/t/data/vball/skyball/jersey/gallery/v0
```

The sampler is deterministic for the same code, inputs, and options. The default seed is `20260606`; override it with `--seed`.
