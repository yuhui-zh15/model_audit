import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import Linear
from utils import LossComputer, computing_subgroup_metrics, subgrouping

CLIP_MODEL = "ViT-B/32"


def train_one_epoch(dataloader, model, optimizer, device="cuda"):
    model.train()
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_one_epoch_dro(dataloader, model, optimizer, loss_fn, device="cuda"):
    model.train()
    for batch in dataloader:
        x, y, gidxs = batch
        x, y, gidxs = x.to(device), y.to(device), gidxs.to(device)
        logits = model(x)
        loss = loss_fn.loss(logits, y, group_idx=gidxs, is_training=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(dataloader, model, device="cuda"):
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
    preds, labels = np.array(preds), np.array(labels)
    acc = np.mean(preds == labels)
    loss = np.mean(losses)
    return {
        "acc": acc,
        "loss": loss,
        "preds": preds,
        "labels": labels,
    }


def train_image_model_fairface(data_path: str, feature_path: str):
    data = [json.loads(line) for line in open(data_path)]
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
    print(montana_race_norm)

    sampled_train_idxs = []
    random.seed(1234)
    for idx in train_idxs:
        race = data[idx]["attributes"]["race"]
        if race in montana_race_norm:
            if random.random() <= montana_race_norm[race]:
                sampled_train_idxs.append(idx)

    # race_distribution = Counter([item["attributes"]["race"] for item in sampled_train_data]).most_common()

    features = F.normalize(torch.load(feature_path))

    train_features = features[sampled_train_idxs]
    train_labels = labels[sampled_train_idxs]
    val_features = features[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Linear(512, 2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch_idx in range(25):
        train_one_epoch(train_dataloader, model, opt)
        metrics = evaluate(val_dataloader, model)
        print(epoch_idx, metrics)

    torch.save(model.state_dict(), "fairface_linear_model.pt")


def train_image_model_dspites(data_path: str, feature_path: str):
    data = [json.loads(line) for line in open(data_path)]
    labels = torch.tensor([item["attributes"]["label"] for item in data])

    all_train_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["color"] == "red"
    ]

    random.seed(1234)
    train_idxs = random.sample(all_train_idxs, int(len(all_train_idxs) * 0.9))
    val_idxs = [idx for idx in range(len(data)) if idx not in train_idxs]
    json.dump([train_idxs, val_idxs], open("train_val_idxs.json", "w"))
    print(len(train_idxs), len(val_idxs))

    features = F.normalize(torch.load(feature_path))

    train_features = features[train_idxs]
    train_labels = labels[train_idxs]
    val_features = features[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Linear(512, 2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch_idx in range(25):
        train_one_epoch(train_dataloader, model, opt)
        metrics = evaluate(val_dataloader, model)
        print(epoch_idx, metrics)

    torch.save(model.state_dict(), "dspites_linear_model.pt")


def train_image_model_waterbird(
    data_path: str, feature_path: str, close_gap: bool, coco_norm: bool = True
):
    data = [json.loads(line) for line in open(data_path)]
    labels = torch.tensor([item["attributes"]["waterbird"] for item in data])

    train_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "train"
    ]
    val_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "val"
    ]

    features = F.normalize(torch.load(feature_path))

    if close_gap:
        if coco_norm:
            coco_features = torch.load("pytorch_cache/features/coco_features_vitb32.pt")
            coco_mean = F.normalize(coco_features["image_features"]).mean(0)
            features = features - coco_mean
        else:
            features = features - features.mean(0)

    train_features = features[train_idxs]
    train_labels = labels[train_idxs]
    val_features = features[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Linear(512, 2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch_idx in range(25):
        train_one_epoch(train_dataloader, model, opt)
        metrics = evaluate(val_dataloader, model)
        print(epoch_idx, metrics)

    torch.save(model.state_dict(), f"waterbird_linear_model_gap{not close_gap}.pt")


def train_image_model_waterbird_dro(data_path: str, feature_path: str):
    data = [json.loads(line) for line in open(data_path)]
    for item in data:
        if (
            item["attributes"]["waterbird"] == 1
            and item["attributes"]["waterplace"] == 1
        ):
            item["attributes"]["group_idx"] = 0
        elif (
            item["attributes"]["waterbird"] == 1
            and item["attributes"]["waterplace"] == 0
        ):
            item["attributes"]["group_idx"] = 1
        elif (
            item["attributes"]["waterbird"] == 0
            and item["attributes"]["waterplace"] == 1
        ):
            item["attributes"]["group_idx"] = 2
        else:
            item["attributes"]["group_idx"] = 3

    labels = torch.tensor([item["attributes"]["waterbird"] for item in data])

    train_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "train"
    ]
    val_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "val"
    ]

    features = F.normalize(torch.load(feature_path))

    train_features = features[train_idxs]
    train_labels = labels[train_idxs]
    train_group_idxs = torch.tensor([item["attributes"]["group_idx"] for item in data])[
        train_idxs
    ]
    val_features = features[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset = TensorDataset(train_features, train_labels, train_group_idxs)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Linear(512, 2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    group_counts = torch.tensor(
        [
            np.sum(
                [
                    item["attributes"]["group_idx"] == i
                    for item in np.array(data)[train_idxs]
                ]
            )
            for i in range(4)
        ]
    ).cuda()
    print(group_counts)
    loss_dro = LossComputer(
        torch.nn.CrossEntropyLoss(reduce=False),
        True,
        0.2,
        0.1,
        np.array([0.0, 0.0, 0.0, 0.0]),
        0,
        0.01,
        False,
        False,
        4,
        group_counts,
        ["ww", "wl", "lw", "ll"],
    )
    for epoch_idx in range(25):
        train_one_epoch_dro(train_dataloader, model, opt, loss_dro)
        metrics = evaluate(val_dataloader, model)
        print(epoch_idx, metrics)

        preds, labels = metrics["preds"], metrics["labels"]
        subgroups = subgrouping(
            list(np.array(data)[val_idxs]), ["waterbird", "waterplace"]
        )
        subgroup_metrics = computing_subgroup_metrics(preds, labels, subgroups)  # type: ignore
        print(subgroup_metrics)

    torch.save(model.state_dict(), "waterbird_linear_model_dro.pt")


def train_image_model_fairface_dro(data_path: str, feature_path: str):
    data = [json.loads(line) for line in open(data_path)]
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
    race2idx = {
        "White": 0,
        "Indian": 1,
        "East Asian": 2,
        "Black": 3,
    }
    for item in data:
        race = item["attributes"]["race"]
        item["attributes"]["group_idx"] = race2idx[race] if race in race2idx else -1

    montana_race_norm = {
        key: montana_race[key] / montana_race["White"] for key in montana_race
    }
    print(montana_race_norm)

    sampled_train_idxs = []
    random.seed(1234)
    for idx in train_idxs:
        race = data[idx]["attributes"]["race"]
        if race in montana_race_norm:
            if random.random() <= montana_race_norm[race]:
                sampled_train_idxs.append(idx)

    # race_distribution = Counter([item["attributes"]["race"] for item in sampled_train_data]).most_common()

    features = F.normalize(torch.load(feature_path))

    train_features = features[sampled_train_idxs]
    train_labels = labels[sampled_train_idxs]
    train_group_idxs = torch.tensor([item["attributes"]["group_idx"] for item in data])[
        sampled_train_idxs
    ]
    val_features = features[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset = TensorDataset(train_features, train_labels, train_group_idxs)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Linear(512, 2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    group_counts = torch.tensor(
        [
            np.sum(
                [
                    item["attributes"]["group_idx"] == i
                    for item in np.array(data)[train_idxs]
                ]
            )
            for i in range(4)
        ]
    ).cuda()
    print(group_counts)
    loss_dro = LossComputer(
        torch.nn.CrossEntropyLoss(reduce=False),
        True,
        0.2,
        0.1,
        np.array([0.0, 0.0, 0.0, 0.0]),
        0,
        0.01,
        False,
        False,
        4,
        group_counts,
        ["ww", "wl", "lw", "ll"],
    )
    for epoch_idx in range(25):
        train_one_epoch_dro(train_dataloader, model, opt, loss_dro)
        metrics = evaluate(val_dataloader, model)
        print(epoch_idx, metrics)

        preds, labels = metrics["preds"], metrics["labels"]
        subgroups = subgrouping(list(np.array(data)[val_idxs]), ["race"])
        subgroup_metrics = computing_subgroup_metrics(preds, labels, subgroups)  # type: ignore
        print(subgroup_metrics)

    torch.save(model.state_dict(), "fairface_linear_model_dro.pt")


if __name__ == "__main__":
    train_image_model_waterbird(
        "../data/Waterbird/processed_attribute_dataset/attributes.jsonl",
        "pytorch_cache/features/waterbird_features_vitb32.pt",
        close_gap=False,
    )
    train_image_model_fairface(
        "../data/FairFace/processed_attribute_dataset/attributes.jsonl",
        "pytorch_cache/features/fairface_features_vitb32.pt",
    )
    train_image_model_dspites(
        "../data/TriangleSquare/processed_attribute_dataset/attributes.jsonl",
        "pytorch_cache/features/trianglesquare_features_vitb32.pt",
    )
    train_image_model_waterbird_dro(
        "../data/Waterbird/processed_attribute_dataset/attributes.jsonl",
        "pytorch_cache/features/waterbird_features_vitb32.pt",
    )
    train_image_model_fairface_dro(
        "../data/FairFace/processed_attribute_dataset/attributes.jsonl",
        "pytorch_cache/features/fairface_features_vitb32.pt",
    )
