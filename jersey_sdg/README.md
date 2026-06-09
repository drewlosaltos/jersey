# Jersey SDG

Synthetic dataset generation for volleyball jersey recognition and player re-identification.

Input data is read from `/mnt/t/data/vball/skyball`. Trial outputs are written under
`jersey_sdg/outputs/`.

Typical trial crop generation:

```bash
conda run -n pt5090new python -m jersey_sdg.cli build-trial \
  --data-root /mnt/t/data/vball/skyball \
  --output /mnt/t/output/jersey_sgd/trial_001
```

## Jersey Recognition Gallery

Before training the jersey recognition model, build a larger crop gallery for
jersey recognition. Do not overwrite `/mnt/t/data/vball/skyball/jersey/gallery/v0`;
that dataset is used by CLIP re-id training. Generate the larger gallery in a
separate output directory with 60 crops per player identity:

```bash
python jersey_gallery/jersey_gallery.py \
  --group-scope match \
  --data-root /mnt/t/data/vball/skyball/unified_dataset \
  --split-file /mnt/t/data/vball/skyball/unified_dataset/splits/v0.json \
  --split-names train val \
  --output /mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/v0_4x \
  --target-count 60
```

This uses the same SkyBall `v0` split/source and gallery recipe as the original
15-crop gallery, but increases `--target-count` from 15 to 60. The default
gallery builder settings are deterministic with seed `20260606`,
`--group-scope match`, `--iou-threshold 0.40`, and `--bbox-expand 0.0`.

Expected outputs:

- `train/manifests/gallery_samples.jsonl`
- `val/manifests/gallery_samples.jsonl`
- `train/reports/summary.json`
- `val/reports/summary.json`

The current `v0_4x` build contains 35,563 train crops across 599 identities and
3,844 validation crops across 65 identities. After this gallery exists, run the
Qwen visibility-label generation over these manifests, then use those labels to
train the uncertainty-aware jersey recognition model.

For the unified dataset import flow, create a versioned gallery that matches the
split version. For example, after creating
`/mnt/t/data/vball/skyball/unified_dataset/splits/v1.json`, build the CLIP-ReID
gallery in `/mnt/t/data/vball/skyball/jersey/gallery/v1`, then build the 4x
jersey-recognition gallery in a separate directory:

```bash
python jersey_gallery/jersey_gallery.py \
  --group-scope match \
  --data-root /mnt/t/data/vball/skyball/unified_dataset \
  --split-file /mnt/t/data/vball/skyball/unified_dataset/splits/v1.json \
  --split-names train val \
  --output /mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/v1_4x \
  --target-count 60
```

When chaining this after detector and CLIP-ReID training, prefer the checked-in
watcher script over a long inline `tmux` command. It waits for the 4x gallery,
waits for CLIP-ReID GPU sessions to finish, starts vLLM/Qwen on GPUs 2,3, labels
train and val, and stops the server when done:

```bash
tmux new-session -d -s qwen_visibility_v1_wait \
  /home/atao/vsdevel/skyball/jersey/jersey_sdg/scripts/run_qwen_visibility_v1_when_ready.sh
```

This avoids a common shell bug where `$!` gets expanded while composing an inline
command instead of after `vllm serve` starts. Check progress in:

- `/mnt/t/output/jersey_sgd/gallery_visibility_4x_v1/logs/qwen_visibility_v1_wait.log`
- `/mnt/t/output/jersey_sgd/gallery_visibility_4x_v1/logs/vllm_qwen.log`

The v1 watcher uses `--tensor-parallel-size 2`, `CUDA_VISIBLE_DEVICES=2,3`,
`--max-model-len 16384`, and 48 labeler workers so RF-DETR can continue on GPUs
0,1. If all four GPUs are free and throughput matters more than concurrent
training, switch back to tensor parallel size 4, all four devices, and 96 workers.

## Qwen Visibility Labeling

After generating the jersey-recognition gallery, auto-label visibility with Qwen.
The labeler asks Qwen whether jersey digits are visible and legible, and marks a
crop as `synthetic_visible=true` only when Qwen says the digits are visible and
the predicted number matches the ground-truth jersey number.

Run the model server from the repository root, not from inside `jersey_sdg`:

```bash
cd /home/atao/vsdevel/skyball/jersey
```

Create the vLLM environment if it does not already exist:

```bash
conda create -y -n vllm5090 python=3.12
conda install -y -n vllm5090 pip
conda run --no-capture-output -n vllm5090 python -m pip install -U pip setuptools wheel
conda run --no-capture-output -n vllm5090 python -m pip install vllm==0.22.1
```

Start Qwen through vLLM on all four RTX 5090 GPUs:

```bash
mkdir -p /mnt/t/output/jersey_sgd/gallery_visibility_4x_v0/logs

tmux new-session -d -s vllm_qwen '
cd /home/atao/vsdevel/skyball/jersey &&
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_ALLREDUCE_USE_FLASHINFER=0 \
conda run --no-capture-output -n vllm5090 \
  vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --served-model-name Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --gpu-memory-utilization 0.72 \
    --max-model-len 32768 \
  > /mnt/t/output/jersey_sgd/gallery_visibility_4x_v0/logs/vllm_qwen.log 2>&1
'
```

Wait for the OpenAI-compatible endpoint to become ready:

```bash
curl http://127.0.0.1:8000/v1/models
tail -f /mnt/t/output/jersey_sgd/gallery_visibility_4x_v0/logs/vllm_qwen.log
```

The `Qwen/Qwen3.6-35B-A3B-FP8` checkpoint declares FP8 quantization in its model
config, and vLLM uses that automatically. `VLLM_USE_FLASHINFER_SAMPLER=0` only
disables the FlashInfer sampler path, which hit a 5090 capability check issue in
our setup. This labeling workload is dominated by image encoding and prefill,
not decode sampling, so disabling that sampler did not limit throughput.

Label the 4x train split:

```bash
conda run --no-capture-output -n vllm5090 python -m jersey_sdg.cli label-gallery-split \
  --gallery-root /mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/v0_4x \
  --split train \
  --output /mnt/t/output/jersey_sgd/gallery_visibility_4x_v0 \
  --endpoint http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --workers 96 \
  --timeout 240 \
  --chunk-size 5000
```

Label the 4x validation split:

```bash
conda run --no-capture-output -n vllm5090 python -m jersey_sdg.cli label-gallery-split \
  --gallery-root /mnt/t/data/vball/skyball/jersey/gallery_jersey_recognition/v0_4x \
  --split val \
  --output /mnt/t/output/jersey_sgd/gallery_visibility_4x_v0 \
  --endpoint http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --workers 96 \
  --timeout 240 \
  --chunk-size 5000
```

The labeling commands are resumable. If the output JSONL already exists, the
labeler skips completed `crop_path` rows and appends only pending rows. Use
`--overwrite` only when intentionally discarding previous labels.

Expected outputs:

- `/mnt/t/output/jersey_sgd/gallery_visibility_4x_v0/manifests/gallery_train_visibility_labels.jsonl`
- `/mnt/t/output/jersey_sgd/gallery_visibility_4x_v0/manifests/gallery_val_visibility_labels.jsonl`
- `/mnt/t/output/jersey_sgd/gallery_visibility_4x_v0/reports/gallery_train_visibility_summary.json`
- `/mnt/t/output/jersey_sgd/gallery_visibility_4x_v0/reports/gallery_val_visibility_summary.json`

For generic manifests such as SoccerNet, use `label-manifest` instead of
`label-gallery-split`, passing `--source-manifest`, `--crop-root`,
`--output-jsonl`, and `--summary-json` explicitly.

Run tests:

```bash
conda run -n pt5090new python -m pytest jersey_sdg/tests
```
