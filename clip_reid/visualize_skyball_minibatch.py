import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from clipreid.skyball import SkyBallGroupedBatchSampler, SkyBallTrainDataset, skyball_dataframe


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit_on_canvas(img: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    scale = min(width / img.shape[1], height / img.shape[0])
    new_w = max(1, int(round(img.shape[1] * scale)))
    new_h = max(1, int(round(img.shape[0] * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def put_label(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (255, 255, 255), -1)
    cv2.putText(
        img,
        text,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one grouped SkyBall training minibatch.")
    parser.add_argument("--gallery-root", default="/mnt/t/data/vball/skyball/jersey/gallery/v0")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="./debug/skyball_grouped_minibatch.jpg")
    parser.add_argument("--batch-size", type=int, default=0, help="Use 0 for one full match/team roster per batch.")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cell-width", type=int, default=180)
    parser.add_argument("--cell-height", type=int, default=240)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = skyball_dataframe(args.gallery_root, args.split)
    shuffle_batch_size = args.batch_size if args.batch_size > 0 else 16
    dataset = SkyBallTrainDataset(df, prob_flip=0.0, shuffle_batch_size=shuffle_batch_size)
    sampler = SkyBallGroupedBatchSampler(
        dataset,
        batch_size=args.batch_size if args.batch_size > 0 else None,
        seed=args.seed,
    )

    batch = None
    for index, candidate in enumerate(sampler):
        if index == args.batch_index:
            batch = candidate
            break
    if batch is None:
        raise IndexError(f"batch-index {args.batch_index} is outside {len(sampler)} batches")

    rows = []
    meta_rows = []
    for item_index in batch:
        query_path = dataset.samples[item_index]
        row = dataset.df.loc[query_path]
        positive_path = np.random.choice(dataset.player_images_other[query_path], 1)[0]

        query_tile = fit_on_canvas(read_rgb(query_path), args.cell_width, args.cell_height)
        positive_tile = fit_on_canvas(read_rgb(positive_path), args.cell_width, args.cell_height)
        label = f"{row['group_id']} #{row['jersey_number']} r{row['sample_rank']}"
        put_label(query_tile, "query " + label)
        put_label(positive_tile, "positive " + label)
        rows.append(np.concatenate([query_tile, positive_tile], axis=1))
        meta_rows.append(row)

    header_h = 44
    width = args.cell_width * 2
    header = np.full((header_h, width, 3), 255, dtype=np.uint8)
    group_ids = sorted({row["group_id"] for row in meta_rows})
    player_count = len({int(row["player"]) for row in meta_rows})
    title = f"SkyBall {args.split} grouped batch {args.batch_index}: {group_ids[0]} | players={player_count}"
    cv2.putText(header, title, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)

    grid = np.concatenate([header, *rows], axis=0)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(out)
    print(f"group_ids={group_ids}")
    print(f"players={player_count}")


if __name__ == "__main__":
    main()
