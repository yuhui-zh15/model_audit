import json
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import Linear
from utils import LossComputer, computing_subgroup_metrics, subgrouping


def train_one_epoch(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
):
    model.train()
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_one_epoch_dro(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossComputer,
    device: str = "cuda",
):
    model.train()
    for batch in dataloader:
        x, y, gidxs = batch
        x, y, gidxs = x.to(device), y.to(device), gidxs.to(device)
        logits = model(x)
        loss = loss_fn.loss(logits, y, group_idx=gidxs, is_training=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: str = "cuda",
) -> dict:
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(y.cpu().tolist())
            losses.append(loss.item())
    preds_np, labels_np = np.array(preds), np.array(labels)
    acc = np.mean(preds_np == labels_np)
    mean_loss = np.mean(losses)
    return {
        "acc": acc,
        "loss": mean_loss,
        "preds": preds_np,
        "labels": labels_np,
    }


def train_image_model(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idxs: list,
    val_idxs: list,
    data: Optional[list] = None,
    fields: Optional[list] = None,
    n_epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    close_gap: bool = False,
    global_mean: Optional[torch.Tensor] = None,
) -> torch.nn.Module:
    assert len(features) == len(
        labels
    ), "Features and labels should have the same length."
    features = F.normalize(features)

    if close_gap:
        if global_mean is not None:
            assert global_mean.shape == features.shape[1:]
            features = features - global_mean
        else:
            features = features - features.mean(0)

    train_features = features[train_idxs]
    train_labels = labels[train_idxs]
    val_features = features[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    d_model = features.shape[1]
    n_classes = int(labels.max().item() + 1)
    model = Linear(d_model, n_classes).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch_idx in range(n_epochs):
        train_one_epoch(train_dataloader, model, opt)
        metrics = evaluate(val_dataloader, model)
        print(epoch_idx, metrics)

        if fields is not None:
            assert (
                data is not None
            ), "Data must be provided to compute subgroup metrics."
            assert len(data) == len(
                features
            ), "Data and features should have the same length."
            preds, labels = metrics["preds"], metrics["labels"]
            subgroups = subgrouping([data[idx] for idx in val_idxs], fields)
            subgroup_metrics = computing_subgroup_metrics(
                preds.tolist(), labels.tolist(), subgroups
            )
            print(sorted(subgroup_metrics.items(), key=lambda x: x[1]))

    return model


def train_image_model_dro(
    features: torch.Tensor,
    labels: torch.Tensor,
    group_idxs: torch.Tensor,
    train_idxs: list,
    val_idxs: list,
    data: Optional[list] = None,
    fields: Optional[list] = None,
    n_epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> torch.nn.Module:
    assert len(features) == len(
        labels
    ), "Features and labels should have the same length."
    features = F.normalize(features)

    train_features = features[train_idxs]
    train_labels = labels[train_idxs]
    train_group_idxs = group_idxs[train_idxs]
    val_features = features[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset = TensorDataset(train_features, train_labels, train_group_idxs)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    d_model = features.shape[1]
    n_classes = int(labels.max().item() + 1)
    model = Linear(d_model, n_classes).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    group_counts = train_group_idxs.bincount().cuda()

    loss_dro = LossComputer(
        torch.nn.CrossEntropyLoss(reduce=False),
        True,
        0.2,
        0.1,
        None,
        0,
        0.01,
        False,
        False,
        len(group_counts),
        group_counts,
        [f"g{i}" for i in range(len(group_counts))],
    )

    for epoch_idx in range(n_epochs):
        train_one_epoch_dro(train_dataloader, model, opt, loss_dro)
        metrics = evaluate(val_dataloader, model)
        print(epoch_idx, metrics)

        if fields is not None:
            assert (
                data is not None
            ), "Data must be provided to compute subgroup metrics."
            assert len(data) == len(
                features
            ), "Data and features should have the same length."
            preds, labels = metrics["preds"], metrics["labels"]
            subgroups = subgrouping([data[idx] for idx in val_idxs], fields)
            subgroup_metrics = computing_subgroup_metrics(
                preds.tolist(), labels.tolist(), subgroups
            )
            print(sorted(subgroup_metrics.items(), key=lambda x: x[1]))

    return model


def train_image_model_waterbird(
    data_path: str,
    feature_path: str,
    close_gap: bool = False,
    coco_norm: bool = True,
    dro: bool = False,
):
    data = [json.loads(line) for line in open(data_path)]
    features = F.normalize(torch.load(feature_path))
    labels = torch.tensor([item["attributes"]["waterbird"] for item in data])

    train_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "train"
    ]
    val_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "val"
    ]

    global_mean = None
    if close_gap and coco_norm:
        coco_features = torch.load("pytorch_cache/features/coco_features_vitb32.pt")
        global_mean = F.normalize(coco_features["image_features"]).mean(0)

    if not dro:
        model = train_image_model(
            features,
            labels,
            train_idxs,
            val_idxs,
            data=data,
            fields=["waterbird", "waterplace"],
            close_gap=close_gap,
            global_mean=global_mean,
        )
    else:
        assert not close_gap, "DRO is not compatible with close gap."
        group2idx = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3,
        }
        group_idxs = torch.tensor(
            [
                group2idx.get(
                    (item["attributes"]["waterbird"], item["attributes"]["waterplace"]),
                    -1,
                )
                for item in data
            ]
        )
        model = train_image_model_dro(
            features,
            labels,
            group_idxs,
            train_idxs,
            val_idxs,
            data=data,
            fields=["waterbird", "waterplace"],
        )

    torch.save(
        model.state_dict(), f"waterbird_linear_model_gap{not close_gap}_dro{dro}.pt"
    )


def train_image_model_fairface(data_path: str, feature_path: str, dro: bool = False):
    data = [json.loads(line) for line in open(data_path)]
    features = F.normalize(torch.load(feature_path))
    labels = torch.tensor(
        [1 if item["attributes"]["gender"] == "Female" else 0 for item in data]
    )

    train_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "train"
    ]
    val_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "val"
    ]

    montana_race = {
        "White": 90.6 / 100,
        "Indian": 6.2 / 100,
        "East Asian": 0.5 / 100,
        "Black": 0.3 / 100,
    }
    montana_race_norm = {
        key: montana_race[key] / montana_race["White"] for key in montana_race
    }

    sampled_train_idxs = []
    random.seed(1234)
    for idx in train_idxs:
        race = data[idx]["attributes"]["race"]
        if race in montana_race_norm:
            if random.random() <= montana_race_norm[race]:
                sampled_train_idxs.append(idx)

    if not dro:
        model = train_image_model(
            features, labels, sampled_train_idxs, val_idxs, data=data, fields=["race"]
        )
    else:
        group2idx = {
            "White": 0,
            "Indian": 1,
            "East Asian": 2,
            "Black": 3,
        }
        group_idxs = torch.tensor(
            [group2idx.get(item["attributes"]["race"], -1) for item in data]
        )
        model = train_image_model_dro(
            features,
            labels,
            group_idxs,
            sampled_train_idxs,
            val_idxs,
            data=data,
            fields=["race"],
        )

    torch.save(model.state_dict(), f"fairface_linear_model_dro{dro}.pt")


def train_image_model_dspites(data_path: str, feature_path: str):
    data = [json.loads(line) for line in open(data_path)]
    features = F.normalize(torch.load(feature_path))
    labels = torch.tensor([item["attributes"]["label"] for item in data])

    all_train_idxs = [
        i
        for i, item in enumerate(data)
        if (item["attributes"]["color"] == "green" and item["attributes"]["label"] == 0)
        or (
            item["attributes"]["color"] == "orange" and item["attributes"]["label"] == 1
        )
    ]

    random.seed(1234)
    train_idxs = random.sample(all_train_idxs, int(len(all_train_idxs) * 0.9))
    val_idxs = [idx for idx in range(len(data)) if idx not in train_idxs]
    json.dump(
        [train_idxs, val_idxs], open("dspites_train_val_idxs.json", "w"), indent=2
    )

    model = train_image_model(
        features, labels, train_idxs, val_idxs, data=data, fields=["color", "label"]
    )
    torch.save(model.state_dict(), "dspites_linear_model.pt")


if __name__ == "__main__":
    train_image_model_waterbird(
        "../data/Waterbird/processed_attribute_dataset/attributes.jsonl",
        "pytorch_cache/features/waterbird_features_vitb32.pt",
    )
    train_image_model_fairface(
        "../data/FairFace/processed_attribute_dataset/attributes.jsonl",
        "pytorch_cache/features/fairface_features_vitb32.pt",
    )
    train_image_model_dspites(
        "../data/TriangleSquare/processed_attribute_dataset/attributes.jsonl",
        "pytorch_cache/features/trianglesquare_features_vitb32.pt",
    )
    train_image_model_waterbird(
        "../data/Waterbird/processed_attribute_dataset/attributes.jsonl",
        "pytorch_cache/features/waterbird_features_vitb32.pt",
        dro=True,
    )
    train_image_model_fairface(
        "../data/FairFace/processed_attribute_dataset/attributes.jsonl",
        "pytorch_cache/features/fairface_features_vitb32.pt",
        dro=True,
    )
