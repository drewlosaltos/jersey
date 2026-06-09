import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from clipreid.evaluator import compute_dist_matrix, compute_scores, predict
from clipreid.loss import ClipLoss
from clipreid.model import OpenClipModel, TimmModel
from clipreid.skyball import (
    SkyBallEvalDataset,
    SkyBallGroupedBatchSampler,
    SkyBallTrainDataset,
    compute_group_restricted_dist_matrix,
    skyball_dataframe,
    skyball_img_groups,
)
from clipreid.trainer import get_scheduler, train
from clipreid.transforms import get_transforms
from clipreid.utils import setup_system


def build_model(args):
    if args.model_backend == "open_clip":
        model = OpenClipModel(args.clip_model, args.clip_pretrained, remove_proj=args.remove_proj)
        img_size = model.get_image_size()
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        model = TimmModel(args.timm_model, pretrained=True)
        img_size = (224, 224)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    return model, img_size, mean, std


def evaluate(model, loader, dataset, device, img_groups, global_gallery=False):
    features = predict(model, loader, device, normalize_features=True, verbose=True)
    if global_gallery:
        dist_matrix, dist_matrix_rerank = compute_dist_matrix(features, dataset.query, dataset.gallery, rerank=True)
    else:
        dist_matrix, dist_matrix_rerank = compute_group_restricted_dist_matrix(
            features,
            dataset.query,
            dataset.gallery,
            img_groups,
            rerank=True,
        )
    print("\nWithout re-ranking:")
    mAP = compute_scores(dist_matrix, dataset.query, dataset.gallery, cmc_scores=True)
    print("\nWith re-ranking:")
    mAP_rerank = compute_scores(dist_matrix_rerank, dataset.query, dataset.gallery, cmc_scores=True)
    return mAP, mAP_rerank


def summarize_batch_sizes(loader):
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is None:
        return None
    try:
        sizes = [len(batch) for batch in batch_sampler]
    except TypeError:
        return None
    if not sizes:
        return None
    return {
        "count": len(sizes),
        "min": int(np.min(sizes)),
        "max": int(np.max(sizes)),
        "mean": float(np.mean(sizes)),
    }


def add_tensorboard_args(parser):
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="TensorBoard log directory. Defaults to <output-dir>/tensorboard.",
    )
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP-ReIdent on SkyBall after upstream training.")
    parser.add_argument("--gallery-root", default="/mnt/t/data/vball/skyball/jersey/gallery/v0")
    parser.add_argument("--checkpoint-start", default="./model/ViT-L-14_openai/fold-1_seed_1/weights_e4.pth")
    parser.add_argument("--output-dir", default="./model/skyball/ViT-L-14_openai/fold-1_seed_1")
    parser.add_argument("--model-backend", choices=["open_clip", "timm"], default="open_clip")
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--timm-model", default="vit_base_patch16_224")
    parser.add_argument("--remove-proj", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="SkyBall grouped train batch size. Use 0 for one full match/team roster per batch.",
    )
    parser.add_argument("--batch-size-eval", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--lr-end", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["polynomial", "cosine", "linear", "constant"], default="polynomial")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--prob-flip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--gpu-ids", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--query-count", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-group-batches",
        action="store_true",
        help="Disable default SkyBall same match/team training batches.",
    )
    parser.add_argument(
        "--global-gallery-eval",
        action="store_true",
        help="Evaluate against every gallery image instead of only the query's match/team group.",
    )
    parser.add_argument(
        "--drop-last-group-batches",
        action="store_true",
        help="Drop smaller same-match/team grouped batches.",
    )
    add_tensorboard_args(parser)
    args = parser.parse_args()

    setup_system(seed=args.seed, cudnn_benchmark=True, cudnn_deterministic=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, out / "train_skyball.py")
    writer = None
    if not args.no_tensorboard:
        tensorboard_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else out / "tensorboard"
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        writer.add_text("config/args", json.dumps(vars(args), indent=2, sort_keys=True), 0)

    model, img_size, mean, std = build_model(args)
    if args.checkpoint_start:
        state = torch.load(args.checkpoint_start, map_location="cpu")
        model.load_state_dict(state, strict=True)

    if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        multi_gpu = True
    else:
        multi_gpu = False
    model = model.to(args.device)

    val_transforms, train_transforms = get_transforms(img_size, mean, std)
    train_df = skyball_dataframe(args.gallery_root, "train", query_count=args.query_count)
    val_df = skyball_dataframe(args.gallery_root, "val", query_count=args.query_count)

    train_dataset = SkyBallTrainDataset(
        train_df,
        image_transforms=train_transforms,
        prob_flip=args.prob_flip,
        shuffle_batch_size=args.batch_size,
    )
    val_dataset = SkyBallEvalDataset(val_df, image_transforms=val_transforms)
    val_img_groups = skyball_img_groups(val_df)

    if args.no_group_batches:
        if args.batch_size <= 0:
            raise ValueError("--no-group-batches requires --batch-size > 0")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        print(f"SkyBall training batches: global unique-player batches, batch_size={args.batch_size}")
    else:
        train_sampler = SkyBallGroupedBatchSampler(
            train_dataset,
            batch_size=args.batch_size if args.batch_size > 0 else None,
            seed=args.seed,
            drop_last=args.drop_last_group_batches,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        batch_label = "full roster" if args.batch_size <= 0 else str(args.batch_size)
        print(f"SkyBall training batches: same match/team, batch_size={batch_label}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_eval,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    loss_function = ClipLoss(loss_function=loss_fn, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(init_scale=2.0**10) if args.mixed_precision else None

    config_for_scheduler = argparse.Namespace(
        epochs=args.epochs,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        lr_end=args.lr_end,
        gradient_accumulation=args.gradient_accumulation,
    )
    scheduler = get_scheduler(config_for_scheduler, optimizer, len(train_loader)) if args.scheduler else None
    batch_stats = summarize_batch_sizes(train_loader)
    if batch_stats is not None:
        print(
            "Train batch stats: "
            f"count={batch_stats['count']} min={batch_stats['min']} "
            f"max={batch_stats['max']} mean={batch_stats['mean']:.2f}"
        )
        if writer is not None:
            for name, value in batch_stats.items():
                writer.add_scalar(f"train_batch/{name}", value, 0)
            writer.add_text("train_batch/stats", json.dumps(batch_stats, indent=2, sort_keys=True), 0)

    print("Zero-shot / start checkpoint SkyBall val:")
    print(f"SkyBall eval gallery scope: {'global' if args.global_gallery_eval else 'same match/team'}")
    mAP, mAP_rerank = evaluate(model, val_loader, val_dataset, args.device, val_img_groups, args.global_gallery_eval)
    if writer is not None:
        writer.add_scalar("val/mAP", mAP, 0)
        writer.add_scalar("val/mAP_rerank", mAP_rerank, 0)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], 0)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch: {epoch}")
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        train_loss = train(
            model,
            dataloader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=args.device,
            scheduler=scheduler,
            scaler=scaler,
            gradient_accumulation=args.gradient_accumulation,
            gradient_clipping=None,
            verbose=True,
            multi_gpu=multi_gpu,
        )
        print(f"Avg. Train Loss = {train_loss:.4f} - Lr = {optimizer.param_groups[0]['lr']:.6f}")
        mAP, mAP_rerank = evaluate(model, val_loader, val_dataset, args.device, val_img_groups, args.global_gallery_eval)
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("val/mAP", mAP, epoch)
            writer.add_scalar("val/mAP_rerank", mAP_rerank, epoch)
            writer.flush()

        checkpoint_path = out / f"weights_e{epoch}.pth"
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)
        if args.no_group_batches:
            train_loader.dataset.shuffle()

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
