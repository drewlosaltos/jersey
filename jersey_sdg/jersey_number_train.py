from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.utils import make_grid

from .io_utils import read_jsonl, write_json


NUM_CLASSES = 100
TENS_CLASSES = 11
ONES_CLASSES = 10


@dataclass(frozen=True)
class JerseySample:
    image_path: Path
    number: int
    tens: int
    ones: int
    visible: bool
    row: dict[str, Any]


def parse_number(value: str) -> int:
    number = int(value)
    if not 0 <= number < NUM_CLASSES:
        raise ValueError(f"jersey number outside supported range 0-99: {value}")
    return number


def digit_labels(number: int) -> tuple[int, int]:
    if number < 10:
        return 10, number
    return number // 10, number % 10


def load_samples(labels_path: Path, crops_root: Path) -> list[JerseySample]:
    samples: list[JerseySample] = []
    for row in read_jsonl(labels_path):
        if row.get("vlm_status") != "ok":
            continue
        number = parse_number(str(row["gt_jersey_number"]))
        tens, ones = digit_labels(number)
        row_path = Path(row["crop_path"])
        samples.append(
            JerseySample(
                image_path=row_path if row_path.is_absolute() else crops_root / row_path,
                number=number,
                tens=tens,
                ones=ones,
                visible=bool(row.get("synthetic_visible")),
                row=row,
            )
        )
    return samples


class JerseyNumberDataset(Dataset[JerseySample]):
    def __init__(self, samples: list[JerseySample], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "number": torch.tensor(sample.number, dtype=torch.long),
            "tens": torch.tensor(sample.tens, dtype=torch.long),
            "ones": torch.tensor(sample.ones, dtype=torch.long),
            "visible": torch.tensor(sample.visible, dtype=torch.bool),
            "crop_path": str(sample.row["crop_path"]),
        }


class EvidentialJerseyModel(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool, head_name: str = "independent") -> None:
        super().__init__()
        if backbone_name.startswith("timm:"):
            model_name = backbone_name.split(":", 1)[1]
            backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            features = backbone.num_features
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Identity()
        elif backbone_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.vit_b_16(weights=weights)
            features = backbone.heads.head.in_features
            backbone.heads.head = nn.Identity()
        else:
            raise ValueError(f"unsupported backbone: {backbone_name}")
        self.backbone = backbone
        self.head_name = head_name
        if head_name == "independent":
            self.number_head = nn.Linear(features, NUM_CLASSES)
        elif head_name == "tda_mb":
            self.position_embedding = nn.Parameter(torch.ones(3, features))
            self.shared_digit_head = nn.Linear(features, ONES_CLASSES)
            self.position_digit_bias = nn.Parameter(torch.zeros(3, ONES_CLASSES))
        else:
            raise ValueError(f"unsupported head: {head_name}")
        self.tens_head = nn.Linear(features, TENS_CLASSES)
        self.ones_head = nn.Linear(features, ONES_CLASSES)

    def _number_scores(self, features: torch.Tensor) -> torch.Tensor:
        if self.head_name == "independent":
            return self.number_head(features)
        single_logits = self.shared_digit_head(features * self.position_embedding[0]) + self.position_digit_bias[0]
        tens_logits = self.shared_digit_head(features * self.position_embedding[1]) + self.position_digit_bias[1]
        ones_logits = self.shared_digit_head(features * self.position_embedding[2]) + self.position_digit_bias[2]
        scores = features.new_empty((features.shape[0], NUM_CLASSES))
        scores[:, :10] = single_logits
        two_digit = []
        for number in range(10, NUM_CLASSES):
            two_digit.append(tens_logits[:, number // 10] + ones_logits[:, number % 10])
        scores[:, 10:] = torch.stack(two_digit, dim=1)
        return scores

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        return {
            "number": F.softplus(self._number_scores(features)) + 1.0,
            "tens": F.softplus(self.tens_head(features)) + 1.0,
            "ones": F.softplus(self.ones_head(features)) + 1.0,
        }


def dirichlet_nll(alpha: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    total = alpha.sum(dim=1)
    log_alpha = torch.log(alpha)
    picked = log_alpha.gather(1, target[:, None]).squeeze(1)
    if label_smoothing <= 0:
        expected = picked
    else:
        num_classes = alpha.shape[1]
        smooth = min(max(label_smoothing, 0.0), 1.0)
        expected = (1.0 - smooth) * picked + smooth * log_alpha.mean(dim=1)
    return (torch.log(total) - expected).mean()


def low_evidence_loss(alpha: torch.Tensor) -> torch.Tensor:
    evidence = alpha - 1.0
    return evidence.square().mean()


def train_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    digit_weight: float,
    label_smoothing: float,
) -> torch.Tensor:
    visible = batch["visible"]
    loss = outputs["number"].new_tensor(0.0)
    if visible.any():
        loss = loss + dirichlet_nll(outputs["number"][visible], batch["number"][visible], label_smoothing)
        loss = loss + digit_weight * dirichlet_nll(outputs["tens"][visible], batch["tens"][visible], label_smoothing)
        loss = loss + digit_weight * dirichlet_nll(outputs["ones"][visible], batch["ones"][visible], label_smoothing)
    if (~visible).any():
        loss = loss + low_evidence_loss(outputs["number"][~visible])
        loss = loss + digit_weight * low_evidence_loss(outputs["tens"][~visible])
        loss = loss + digit_weight * low_evidence_loss(outputs["ones"][~visible])
    return loss


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "number": torch.stack([item["number"] for item in batch]),
        "tens": torch.stack([item["tens"] for item in batch]),
        "ones": torch.stack([item["ones"] for item in batch]),
        "visible": torch.stack([item["visible"] for item in batch]),
        "crop_path": [item["crop_path"] for item in batch],
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    visible_total = 0
    visible_correct = 0
    visible_top5 = 0
    all_visible: list[int] = []
    all_conf: list[float] = []
    uncertainty_visible: list[float] = []
    uncertainty_hidden: list[float] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["number"].to(device, non_blocking=True)
        visible = batch["visible"].to(device, non_blocking=True)
        alpha = model(images)["number"]
        probs = alpha / alpha.sum(dim=1, keepdim=True)
        pred = probs.argmax(dim=1)
        conf = probs.max(dim=1).values
        uncertainty = NUM_CLASSES / alpha.sum(dim=1)
        if visible.any():
            visible_total += int(visible.sum().item())
            visible_correct += int((pred[visible] == targets[visible]).sum().item())
            visible_top5 += int(
                (probs.topk(5, dim=1).indices[visible] == targets[visible, None]).any(dim=1).sum().item()
            )
            uncertainty_visible.extend(uncertainty[visible].detach().cpu().tolist())
        if (~visible).any():
            uncertainty_hidden.extend(uncertainty[~visible].detach().cpu().tolist())
        all_visible.extend(visible.detach().cpu().int().tolist())
        all_conf.extend(conf.detach().cpu().tolist())

    visible_acc = visible_correct / visible_total if visible_total else 0.0
    visible_top5_acc = visible_top5 / visible_total if visible_total else 0.0
    try:
        visibility_auc = roc_auc_score(all_visible, all_conf)
    except ValueError:
        visibility_auc = float("nan")
    return {
        "visible_acc": visible_acc,
        "visible_top5_acc": visible_top5_acc,
        "visible_total": float(visible_total),
        "mean_uncertainty_visible": float(np.mean(uncertainty_visible)) if uncertainty_visible else float("nan"),
        "mean_uncertainty_non_visible": float(np.mean(uncertainty_hidden)) if uncertainty_hidden else float("nan"),
        "visibility_conf_auc": float(visibility_auc),
    }


def make_transforms(train: bool, image_size: int) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def tensor_to_display_image(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = (tensor.detach().cpu() * std + mean).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def _sample_diverse_examples(
    samples: list[JerseySample],
    max_examples: int,
    rng: random.Random,
) -> list[JerseySample]:
    grouped: dict[str, list[JerseySample]] = defaultdict(list)
    for sample in samples:
        key = str(sample.image_path.parent)
        grouped[key].append(sample)

    groups = list(grouped.values())
    rng.shuffle(groups)
    chosen: list[JerseySample] = []
    for group in groups:
        chosen.append(rng.choice(group))
        if len(chosen) >= max_examples:
            break

    if len(chosen) < max_examples:
        chosen_ids = {id(sample) for sample in chosen}
        remaining = [sample for sample in samples if id(sample) not in chosen_ids]
        chosen.extend(rng.sample(remaining, min(max_examples - len(chosen), len(remaining))))

    rng.shuffle(chosen)
    return chosen


@torch.no_grad()
def add_validation_examples(
    writer: SummaryWriter,
    model: nn.Module,
    samples: list[JerseySample],
    transform: transforms.Compose,
    device: torch.device,
    epoch: int,
    max_examples: int,
    seed: int,
) -> None:
    model.eval()
    visible = [sample for sample in samples if sample.visible]
    hidden = [sample for sample in samples if not sample.visible]
    rng = random.Random(seed + epoch * 7919 + int(time.time() * 1000))
    target_visible = max_examples // 2
    target_hidden = max_examples - target_visible
    chosen = _sample_diverse_examples(visible, target_visible, rng)
    chosen.extend(_sample_diverse_examples(hidden, target_hidden, rng))
    if len(chosen) < max_examples:
        chosen_ids = {id(sample) for sample in chosen}
        remaining = [sample for sample in samples if id(sample) not in chosen_ids]
        chosen.extend(_sample_diverse_examples(remaining, max_examples - len(chosen), rng))
    rng.shuffle(chosen)
    if not chosen:
        return

    tensors = []
    display_images = []
    for sample in chosen:
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            tensors.append(transform(image))
            display_images.append(tensor_to_display_image(tensors[-1]))
    batch = torch.stack(tensors).to(device)
    alpha = model(batch)["number"]
    probs = alpha / alpha.sum(dim=1, keepdim=True)
    pred = probs.argmax(dim=1).detach().cpu().tolist()
    conf = probs.max(dim=1).values.detach().cpu().tolist()
    uncertainty = (NUM_CLASSES / alpha.sum(dim=1)).detach().cpu().tolist()

    positive_uncertainty_threshold = 0.7
    positive_prob_threshold = 0.5
    border_px = 4
    panels = []
    for image, sample, p, c, u in zip(display_images, chosen, pred, conf, uncertainty):
        is_positive = u < positive_uncertainty_threshold and c > positive_prob_threshold
        canvas = Image.new("RGB", (image.width + border_px * 2, image.height + 56 + border_px * 2), "white")
        if is_positive:
            canvas.paste("green", (0, 0, canvas.width, canvas.height))
            canvas.paste("white", (border_px, border_px, canvas.width - border_px, canvas.height - border_px))
        canvas.paste(image, (border_px, border_px))
        line1 = f"GT #{sample.number} | PRED #{p} | visible_label={int(sample.visible)}"
        line2 = f"pred_prob={c:.2f} | dirichlet_uncertainty={u:.2f} | positive={int(is_positive)}"
        # Default bitmap font keeps this dependency-free.
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        text_y = border_px + image.height + 4
        draw.text((border_px + 4, text_y), line1, fill="black", font=font)
        draw.text((border_px + 4, text_y + 18), line2, fill="black", font=font)
        panels.append(transforms.ToTensor()(canvas))
    writer.add_image("val_examples/GT_vs_PRED_with_uncertainty", make_grid(panels, nrow=4), epoch)


def run_training(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_crops_root = Path(args.train_crops_root) if args.train_crops_root else Path(args.gallery_root) / "train" / "crops"
    val_crops_root = Path(args.val_crops_root) if args.val_crops_root else Path(args.gallery_root) / "val" / "crops"
    train_samples = load_samples(Path(args.train_labels), train_crops_root)
    val_samples = load_samples(Path(args.val_labels), val_crops_root)
    train_loader = DataLoader(
        JerseyNumberDataset(train_samples, make_transforms(True, args.image_size)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    val_loader = DataLoader(
        JerseyNumberDataset(val_samples, make_transforms(False, args.image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    model = EvidentialJerseyModel(args.backbone, pretrained=not args.no_pretrained, head_name=args.head).to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    best_acc = -math.inf
    history: list[dict[str, Any]] = []
    start_time = time.time()
    writer = SummaryWriter(log_dir=str(output / "tensorboard"))
    writer.add_text("config/backbone", args.backbone, 0)
    writer.add_text("config/output", str(output), 0)
    writer.add_text(
        "legend/validation_images",
        (
            "GT is the ground-truth jersey number from the gallery identity. "
            "PRED is the model's argmax jersey number. "
            "visible_label=1 means Qwen read the same number as GT, so the crop is treated as visible/legible. "
            "pred_prob is the model's expected Dirichlet class probability for PRED. "
            "dirichlet_uncertainty = 100 / total_evidence; higher means the model has less evidence and should be less trusted."
        ),
        0,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            moved = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if key in {"number", "tens", "ones", "visible"}
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda" and args.amp):
                outputs = model(images)
                loss = train_loss(outputs, moved, args.digit_weight, args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "lr": scheduler.get_last_lr()[0],
            **metrics,
            "elapsed_s": round(time.time() - start_time, 1),
        }
        history.append(row)
        print(row, flush=True)
        writer.add_scalar("train/loss", row["train_loss"], epoch)
        writer.add_scalar("train/lr", row["lr"], epoch)
        for key in [
            "visible_acc",
            "visible_top5_acc",
            "mean_uncertainty_visible",
            "mean_uncertainty_non_visible",
            "visibility_conf_auc",
        ]:
            writer.add_scalar(f"val/{key}", row[key], epoch)
        if epoch == 1 or epoch == args.epochs or epoch % args.tb_image_every == 0:
            add_validation_examples(
                writer,
                model,
                val_samples,
                make_transforms(False, args.image_size),
                device,
                epoch,
                args.tb_examples,
                args.seed,
            )
        if metrics["visible_acc"] > best_acc:
            best_acc = metrics["visible_acc"]
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "model": model_to_save.state_dict(),
                    "args": vars(args),
                    "metrics": row,
                    "num_classes": NUM_CLASSES,
                    "tens_classes": TENS_CLASSES,
                    "ones_classes": ONES_CLASSES,
                    "head": args.head,
                },
                output / "best.pt",
            )
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({"model": model_to_save.state_dict(), "args": vars(args), "metrics": row}, output / "last.pt")
        write_json(output / "history.json", {"epochs": history})
        writer.flush()

    write_json(
        output / "summary.json",
        {
            "best_visible_acc": best_acc,
            "epochs": args.epochs,
            "train_samples": len(train_samples),
            "train_visible": sum(sample.visible for sample in train_samples),
            "val_samples": len(val_samples),
            "val_visible": sum(sample.visible for sample in val_samples),
            "output": str(output),
            "backbone": args.backbone,
            "head": args.head,
            "tensorboard": str(output / "tensorboard"),
        },
    )
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery-root", default="/mnt/t/data/vball/skyball/jersey/gallery/v0")
    parser.add_argument("--train-crops-root", default=None)
    parser.add_argument("--val-crops-root", default=None)
    parser.add_argument(
        "--train-labels",
        default="/mnt/t/output/jersey_sgd/gallery_visibility_v0/manifests/gallery_train_visibility_labels.jsonl",
    )
    parser.add_argument(
        "--val-labels",
        default="/mnt/t/output/jersey_sgd/gallery_visibility_v0/manifests/gallery_val_visibility_labels.jsonl",
    )
    parser.add_argument("--output", default="/mnt/t/output/trn/jersey_recognition/uncertainty_v0")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--head", default="independent", choices=["independent", "tda_mb"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--digit-weight", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--tb-image-every", type=int, default=1)
    parser.add_argument("--tb-examples", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
