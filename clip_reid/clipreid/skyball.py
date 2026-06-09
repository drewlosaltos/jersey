import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .evaluator import compute_dist_matrix


def hflip_image(img):
    if isinstance(img, np.ndarray):
        return np.ascontiguousarray(img[:, ::-1])
    return transforms.functional.hflip(img)


def skyball_dataframe(
    gallery_root: str | Path,
    split: str,
    query_count: int = 1,
    min_images_per_identity: int = 2,
) -> pd.DataFrame:
    root = Path(gallery_root)
    manifest_path = root / split / "manifests" / "gallery_samples.jsonl"
    crop_root = root / split / "crops"
    rows = [json.loads(line) for line in manifest_path.open()]

    by_identity: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_identity[row["identity_id"]].append(row)

    player_map = {identity_id: idx for idx, identity_id in enumerate(sorted(by_identity))}
    output_rows = []
    for identity_id, identity_rows in sorted(by_identity.items()):
        identity_rows = sorted(identity_rows, key=lambda row: int(row["sample_rank"]))
        if len(identity_rows) < min_images_per_identity:
            continue
        for idx, row in enumerate(identity_rows):
            group_id = f"{row['gallery_entity_id']}__{row['stable_team_id']}"
            output_rows.append(
                {
                    "img_id": str(crop_root / row["crop_path"]),
                    "player": player_map[identity_id],
                    "identity_id": identity_id,
                    "group_id": group_id,
                    "gallery_entity_id": row["gallery_entity_id"],
                    "img_type": "q" if idx < query_count else "g",
                    "split": split,
                    "sample_rank": int(row["sample_rank"]),
                    "match_id": row["match_id"],
                    "play_id": row["play_id"],
                    "team_side": row["team_side"],
                    "stable_team_id": row["stable_team_id"],
                    "jersey_number": row["jersey_number"],
                }
            )
    return pd.DataFrame(output_rows)


class SkyBallEvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_transforms=None):
        self.df = df.set_index("img_id")
        self.image_transforms = image_transforms
        self.images = self.df.index.values.tolist()

        self.query = []
        self.gallery = []
        self.all = []

        for img_id in self.images:
            player = int(self.df.loc[img_id]["player"])
            img_type = self.df.loc[img_id]["img_type"]
            self.all.append((img_id, player, -1))
            if img_type == "q":
                self.query.append((img_id, player, 0))
            else:
                self.gallery.append((img_id, player, 1))

    def __getitem__(self, index):
        img_id = self.images[index]
        img = cv2.imread(img_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.image_transforms:
            img = self.image_transforms(image=img)["image"]

        player = int(self.df.loc[img_id]["player"])
        img_type = 0 if self.df.loc[img_id]["img_type"] == "q" else 1
        return img, img_id, player, img_type

    def __len__(self):
        return len(self.images)


class SkyBallTrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_transforms=None,
        prob_flip: float = 0.5,
        shuffle_batch_size: int = 16,
    ):
        self.df = df.set_index("img_id")
        self.image_transforms = image_transforms
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.images = self.df.index.values.tolist()

        self.player_images = defaultdict(list)
        for img_id in self.images:
            row = self.df.loc[img_id]
            player = int(row["player"])
            self.player_images[player].append(img_id)

        self.player_images_other = {}
        for img_id in self.images:
            player = self.df.loc[img_id]["player"]
            other_images = copy.deepcopy(self.player_images[player])
            other_images.remove(img_id)
            self.player_images_other[img_id] = np.array(other_images)

        self.samples = copy.deepcopy(self.images)
        self.group_player_indices = self._build_group_player_indices()
        self.shuffle()

    def __getitem__(self, index):
        img_id_query = self.samples[index]
        img_query = cv2.imread(img_id_query)
        img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)

        if self.image_transforms:
            img_query = self.image_transforms(image=img_query)["image"]

        img_id_gallery = np.random.choice(self.player_images_other[img_id_query], 1)[0]
        img_gallery = cv2.imread(img_id_gallery)
        img_gallery = cv2.cvtColor(img_gallery, cv2.COLOR_BGR2RGB)

        if self.image_transforms:
            img_gallery = self.image_transforms(image=img_gallery)["image"]

        player = torch.tensor(int(self.df.loc[img_id_query]["player"]), dtype=torch.long)

        if np.random.random() < self.prob_flip:
            img_query = hflip_image(img_query)
            img_gallery = hflip_image(img_gallery)

        return img_query, img_gallery, player

    def __len__(self):
        return len(self.samples)

    def _build_group_player_indices(self):
        group_player_indices = defaultdict(lambda: defaultdict(list))
        for index, img_id in enumerate(self.images):
            row = self.df.loc[img_id]
            group_player_indices[row["group_id"]][int(row["player"])].append(index)
        return group_player_indices

    def shuffle(self):
        img_ids_select = copy.deepcopy(self.images)
        random.shuffle(img_ids_select)

        batches = []
        players_batch = set()
        break_counter = 0

        while True:
            if len(img_ids_select) > 0:
                img_id = img_ids_select.pop(0)
                player = self.df.loc[img_id]["player"]

                if player not in players_batch:
                    players_batch.add(player)
                    batches.append(img_id)
                    break_counter = 0
                else:
                    img_ids_select.append(img_id)
                    break_counter += 1

                if break_counter >= 10:
                    break
            else:
                break

            if len(players_batch) >= self.shuffle_batch_size:
                players_batch = set()

        self.samples = batches


class SkyBallGroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: SkyBallTrainDataset,
        batch_size: int | None = None,
        seed: int = 1,
        drop_last: bool = False,
    ):
        if batch_size is not None and batch_size == 0:
            batch_size = None
        if batch_size is not None and batch_size < 2:
            raise ValueError("Grouped SkyBall batches need batch_size >= 2")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.dataset.samples = copy.deepcopy(self.dataset.images)
        self.dataset.group_player_indices = self.dataset._build_group_player_indices()

        self.group_player_indices = {
            group_id: {player: list(indices) for player, indices in player_indices.items()}
            for group_id, player_indices in dataset.group_player_indices.items()
            if len(player_indices) >= 2
        }
        self.group_jersey_players = self._build_group_jersey_players()
        if not self.group_player_indices:
            raise ValueError("No SkyBall match/team groups have enough players for grouped training")

    def _build_group_jersey_players(self):
        group_jersey_players = defaultdict(lambda: defaultdict(list))
        for group_id, player_indices in self.group_player_indices.items():
            for player_id, indices in player_indices.items():
                img_id = self.dataset.images[indices[0]]
                jersey_number = str(self.dataset.df.loc[img_id]["jersey_number"])
                group_jersey_players[group_id][jersey_number].append(player_id)
        return group_jersey_players

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []

        group_ids = list(self.group_player_indices)
        rng.shuffle(group_ids)
        for group_id in group_ids:
            player_ids = [
                rng.choice(players)
                for _, players in sorted(self.group_jersey_players[group_id].items())
            ]
            rng.shuffle(player_ids)
            if self.batch_size is None:
                player_chunks = [player_ids]
            else:
                player_chunks = [
                    player_ids[start : start + self.batch_size]
                    for start in range(0, len(player_ids), self.batch_size)
                ]
            for chunk in player_chunks:
                if self.batch_size is not None and len(chunk) < self.batch_size and self.drop_last:
                    continue
                batch = [
                    rng.choice(self.group_player_indices[group_id][player_id])
                    for player_id in chunk
                ]
                if len(batch) >= 2:
                    batches.append(batch)

        rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        if self.batch_size is None:
            return len(self.group_player_indices)

        count = 0
        for group_id in self.group_player_indices:
            jersey_count = len(self.group_jersey_players[group_id])
            full_batches = jersey_count // self.batch_size
            if not self.drop_last and jersey_count % self.batch_size >= 2:
                full_batches += 1
            count += full_batches
        return count


def skyball_img_groups(df: pd.DataFrame) -> dict[str, str]:
    return dict(zip(df["img_id"], df["group_id"]))


def compute_group_restricted_dist_matrix(
    features,
    query,
    gallery,
    img_groups: dict[str, str],
    rerank: bool = True,
):
    dist_matrix = np.full((len(query), len(gallery)), 1.0e6, dtype=np.float32)
    dist_matrix_rerank = np.full((len(query), len(gallery)), 1.0e6, dtype=np.float32) if rerank else None

    query_by_group = defaultdict(list)
    gallery_by_group = defaultdict(list)
    for index, (img_id, _, _) in enumerate(query):
        query_by_group[img_groups[img_id]].append(index)
    for index, (img_id, _, _) in enumerate(gallery):
        gallery_by_group[img_groups[img_id]].append(index)

    for group_id in sorted(query_by_group):
        query_indices = query_by_group[group_id]
        gallery_indices = gallery_by_group.get(group_id, [])
        if not gallery_indices:
            continue

        query_group = [query[index] for index in query_indices]
        gallery_group = [gallery[index] for index in gallery_indices]
        group_result = compute_dist_matrix(
            features,
            query_group,
            gallery_group,
            rerank=rerank,
        )
        if rerank:
            group_dist, group_dist_rerank = group_result
        else:
            group_dist, group_dist_rerank = group_result, None
        group_dist = np.asarray(group_dist, dtype=np.float32)
        dist_matrix[np.ix_(query_indices, gallery_indices)] = group_dist
        if rerank and group_dist_rerank is not None:
            dist_matrix_rerank[np.ix_(query_indices, gallery_indices)] = np.asarray(
                group_dist_rerank,
                dtype=np.float32,
            )

    return dist_matrix, dist_matrix_rerank
