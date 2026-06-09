import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from clipreid.evaluator import compute_dist_matrix, compute_scores, predict
from clipreid.model import OpenClipModel, TimmModel
from clipreid.skyball import (
    SkyBallEvalDataset,
    compute_group_restricted_dist_matrix,
    skyball_dataframe,
    skyball_img_groups,
)
from clipreid.transforms import get_transforms


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLIP-ReIdent on SkyBall gallery manifests.")
    parser.add_argument("--gallery-root", default="/mnt/t/data/vball/skyball/jersey/gallery/v0")
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", default="./model/ViT-L-14_openai/fold-1_seed_1/weights_e4.pth")
    parser.add_argument("--model-backend", choices=["open_clip", "timm"], default="open_clip")
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--timm-model", default="vit_base_patch16_224")
    parser.add_argument("--remove-proj", action="store_true", default=True)
    parser.add_argument("--query-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--global-gallery",
        action="store_true",
        help="Evaluate against every gallery image instead of only the query's match/team group.",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    model, img_size, mean, std = build_model(args)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(args.device)

    val_transforms, _ = get_transforms(img_size, mean, std)
    df = skyball_dataframe(args.gallery_root, args.split, query_count=args.query_count)
    dataset = SkyBallEvalDataset(df, image_transforms=val_transforms)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    print(f"SkyBall split: {args.split}")
    print(f"Images: {len(dataset)} | Query: {len(dataset.query)} | Gallery: {len(dataset.gallery)}")
    print(f"Identities: {df['player'].nunique()}")
    print(f"Gallery scope: {'global' if args.global_gallery else 'same match/team'}")

    features = predict(model, loader, args.device, normalize_features=True, verbose=True)
    if args.global_gallery:
        dist_result = compute_dist_matrix(
            features,
            dataset.query,
            dataset.gallery,
            rerank=not args.no_rerank,
        )
        if args.no_rerank:
            dist_matrix, dist_matrix_rerank = dist_result, None
        else:
            dist_matrix, dist_matrix_rerank = dist_result
    else:
        dist_matrix, dist_matrix_rerank = compute_group_restricted_dist_matrix(
            features,
            dataset.query,
            dataset.gallery,
            skyball_img_groups(df),
            rerank=not args.no_rerank,
        )

    print("\nwithout re-ranking:")
    mAP = compute_scores(dist_matrix, dataset.query, dataset.gallery, cmc_scores=True)
    results = {
        "split": args.split,
        "gallery_scope": "global" if args.global_gallery else "same_match_team",
        "mAP": float(mAP),
        "query": len(dataset.query),
        "gallery": len(dataset.gallery),
    }

    if dist_matrix_rerank is not None:
        print("\nwith re-ranking:")
        mAP_rerank = compute_scores(dist_matrix_rerank, dataset.query, dataset.gallery, cmc_scores=True)
        results["mAP_rerank"] = float(mAP_rerank)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
