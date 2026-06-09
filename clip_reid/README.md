# SkyBall CLIP-ReID

Fine-tune CLIP-ReIdent on SkyBall jersey galleries.

## Data

Default SkyBall gallery root:

```bash
/mnt/t/data/vball/skyball/jersey/gallery/v0
```

The SkyBall loader reads:

```bash
{train,val}/manifests/gallery_samples.jsonl
{train,val}/crops/
```

## Train

Recommended baseline:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_skyball.py \
  --epochs 20 \
  --lr 1e-5 \
  --lr-end 5e-7 \
  --warmup-epochs 0.5 \
  --checkpoint-start ./model/ViT-L-14_openai/fold-1_seed_1/weights_e4.pth \
  --output-dir /mnt/t/output/trn/clip_reid/skyball_roster_v0_vitl14_lr1e-5_e20_seed1
```

Training defaults to full same-match/team roster batches: one sampled image per jersey/player. Use `--batch-size N` only to chunk rosters.

TensorBoard:

```bash
tensorboard --logdir /mnt/t/output/trn/clip_reid --bind_all
```

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_skyball.py \
  --split val \
  --checkpoint /mnt/t/output/trn/clip_reid/skyball_roster_v0_vitl14_lr1e-5_e20_seed1/best_by_rerank.pth
```

Evaluation defaults to same match/team gallery scope. Use `--global-gallery` only for ablations.

## Debug Batches

```bash
python visualize_skyball_minibatch.py \
  --output ./debug/skyball_grouped_minibatch.jpg
```

## Best Model

Recommended checkpoint:

```bash
/mnt/t/output/trn/clip_reid/skyball_roster_v0_vitl14_lr1e-5_e20_seed1/weights_e11.pth
```

Symlink:

```bash
/mnt/t/output/trn/clip_reid/skyball_roster_v0_vitl14_lr1e-5_e20_seed1/best_by_rerank.pth
```

SkyBall val, same match/team gallery, reranked:

```bash
mAP 98.39
rank-1 95.71
rank-5 97.69
rank-10 98.46
```

The absolute best completed mAP was `98.43` from `skyball_roster_v0_vitl14_lr1e-5_e10_seed1/weights_e9.pth`, but the 20-epoch epoch-11 model has better rank metrics and is the recommended checkpoint.

## v1 Data Check

The unified `v1` gallery improves plain embedding quality, but the `v1` val
gallery is harder/different enough that rerank mAP is lower than the old `v0`
rerank result. To separate model quality from validation-set composition, the
new `v1` checkpoints were also evaluated on the old `gallery/v0` val split.

Old `v0` best on `gallery/v0` val:

```text
plain mAP   95.28
rerank mAP  98.39
```

`v1` checkpoints evaluated on `gallery/v0` val:

```text
checkpoint              plain mAP   rerank mAP
v1 e20 weights_e10       97.03       99.14
v1 e20 weights_e20       97.25       99.07
v1 e10 weights_e10       97.26       98.61
```

Conclusion: the `v1`-trained models are not worse than the old model. They
transfer back to `v0` val better than the old `v0` best, so the rerank drop on
`v1` val is likely caused by the new validation/gallery composition rather than
a model regression.
