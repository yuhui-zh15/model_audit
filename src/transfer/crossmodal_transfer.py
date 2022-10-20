import json
import random
from typing import Dict, Tuple

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score  # type: ignore
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange  # type: ignore

import wandb


def train_one_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    multilabel: bool = True,
    device: str = "cuda",
):
    model.train()
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if multilabel:
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
        else:
            loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(
    dataloader: DataLoader,
    model: torch.nn.Module,
    multilabel: bool = True,
    device: str = "cuda",
) -> Dict:
    model.eval()
    preds, labels, losses = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if multilabel:
                loss = F.binary_cross_entropy_with_logits(logits, y.float())
                preds.extend((logits > 0).float().cpu().tolist())
            else:
                loss = F.cross_entropy(logits, y)
                preds.extend(logits.argmax(1).cpu().tolist())

            labels.extend(y.cpu().tolist())
            losses.append(loss.item())

    preds = np.array(preds)  # type: ignore
    labels = np.array(labels)  # type: ignore
    loss = np.mean(losses)  # type: ignore
    if multilabel:
        micro_f1 = f1_score(labels, preds, average="micro")
        macro_f1 = f1_score(labels, preds, average="macro")
        weighted_f1 = f1_score(labels, preds, average="weighted")
        samples_f1 = f1_score(labels, preds, average="samples")
        # baccs = [balanced_accuracy_score(preds[:, i], labels[:, i]) for i in range(80)]
        # bacc = np.mean(baccs)
        return {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "samples_f1": samples_f1,
            # "bacc": bacc,
            "loss": loss,
            "preds": preds,
        }
    else:
        acc = np.mean(preds == labels)
        return {
            "acc": acc,
            "loss": loss,
            "preds": preds,
        }


def close_the_gap(
    img_features: torch.Tensor, txt_features: torch.Tensor, method: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if method == "centering":
        img_features = img_features - img_features.mean(0)
        txt_features = txt_features - txt_features.mean(0)
        img_features = F.normalize(img_features)
        txt_features = F.normalize(txt_features)
    elif method == "centering_norenorm":
        img_features = img_features - img_features.mean(0)
        txt_features = txt_features - txt_features.mean(0)
    elif method == "normcentering_norenorm":
        img_features = img_features - F.normalize(img_features.mean(0), dim=0)
        txt_features = txt_features - F.normalize(txt_features.mean(0), dim=0)
    elif method == "globalbn":
        img_features = img_features - img_features.mean(0)
        txt_features = txt_features - txt_features.mean(0)
        img_features /= img_features.std(0)
        txt_features /= txt_features.std(0)
        img_features = F.normalize(img_features)
        txt_features = F.normalize(txt_features)
    elif method == "original":
        pass
    else:
        raise ValueError("Unknown centering method")
    return img_features, txt_features


def get_optimizer(model: torch.nn.Module, optimizer_type: str) -> torch.optim.Optimizer:
    optimizer: torch.optim.Optimizer
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=0,
        )
    elif optimizer_type == "sgddecay":
        optimizer = torch.optim.SGD(
            [  # type: ignore
                {"params": model.weight, "weight_decay": 1e-4},
                {"params": model.bias, "weight_decay": 0},
            ],
            lr=1e-2,
            momentum=0.9,
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            weight_decay=0,
        )
    elif optimizer_type == "adamdecay":
        optimizer = torch.optim.Adam(
            [  # type: ignore
                {"params": model.weight, "weight_decay": 1e-4},
                {"params": model.bias, "weight_decay": 0},
            ],
            lr=1e-3,
        )
    else:
        raise ValueError("invalid argument")
    return optimizer


def load_coco(
    gap_method: str,
    feature_path: str = "features/coco_features_vitb32.pt",
    label_path: str = "../../../data/COCO/processed_attribute_dataset/attributes.jsonl",
) -> Tuple:
    features = torch.load(feature_path)

    img_features = F.normalize(features["image_features"])
    txt_features = F.normalize(features["text_features"])
    img_features, txt_features = close_the_gap(img_features, txt_features, gap_method)
    labels = features["labels"]

    data = [json.loads(line) for line in open(label_path)]
    train_idxs = [
        i for i in range(len(data)) if data[i]["attributes"]["split"] == "train"
    ]
    val_idxs = [i for i in range(len(data)) if data[i]["attributes"]["split"] == "val"]

    img_features_train = img_features[train_idxs]
    img_labels_train = labels[train_idxs]
    txt_features_train = txt_features[train_idxs]
    txt_labels_train = labels[train_idxs]

    img_features_val = img_features[val_idxs]
    img_labels_val = labels[val_idxs]
    txt_features_val = txt_features[val_idxs]
    txt_labels_val = labels[val_idxs]
    return (
        img_features_train,
        img_labels_train,
        txt_features_train,
        txt_labels_train,
        img_features_val,
        img_labels_val,
        txt_features_val,
        txt_labels_val,
    )


def split_features(features: torch.Tensor, labels: torch.Tensor) -> Tuple:
    random.seed(1234)
    N, _ = features.shape
    N_train = int(N * 0.8)
    train_idxs = sorted(random.sample(range(N), N_train))
    val_idxs = [i for i in range(N) if i not in train_idxs]
    train_features = features[train_idxs, :]
    val_features = features[val_idxs, :]
    train_labels = labels[train_idxs]
    val_labels = labels[val_idxs]
    return train_features, train_labels, val_features, val_labels


def load_imagenet(
    gap_method: str,
    feature_path: str = "features/imagenet_features_vitb32.pt",
):
    features = torch.load(feature_path)

    img_features = F.normalize(features["image_features"])
    txt_features = F.normalize(features["text_features"])
    img_features, txt_features = close_the_gap(img_features, txt_features, gap_method)

    img_labels = features["image_labels"]
    txt_labels = features["text_labels"]

    (
        img_features_train,
        img_labels_train,
        img_features_val,
        img_labels_val,
    ) = split_features(img_features, img_labels)
    (
        txt_features_train,
        txt_labels_train,
        txt_features_val,
        txt_labels_val,
    ) = split_features(txt_features, txt_labels)

    return (
        img_features_train,
        img_labels_train,
        txt_features_train,
        txt_labels_train,
        img_features_val,
        img_labels_val,
        txt_features_val,
        txt_labels_val,
    )


@click.command()
@click.option("--dataset", default="coco", help="coco/imagenet.")
@click.option("--model_type", default="linear", help="linear/mlp/prototype.")
@click.option("--training_modality", default="image", help="image/text.")
@click.option("--optimizer_type", default="adam", help="adam/sgd.")
@click.option("--gap_method", default="original", help="original/centering.")
@click.option("--n_epochs", default=25, help="number of epochs.")
def main(
    dataset: str,
    model_type: str,
    training_modality: str,
    optimizer_type: str,
    gap_method: str,
    n_epochs: int,
):

    n_class = {
        "coco": 80,
        "imagenet": 1000,
    }[dataset]

    (
        img_features_train,
        img_labels_train,
        txt_features_train,
        txt_labels_train,
        img_features_val,
        img_labels_val,
        txt_features_val,
        txt_labels_val,
    ) = (
        load_coco(gap_method) if dataset == "coco" else load_imagenet(gap_method)
    )
    n_dim = img_features_val.shape[1]

    if model_type == "linear":
        model = nn.Linear(n_dim, n_class).cuda()
    elif model_type == "mlp":
        model = nn.Sequential(  # type: ignore
            nn.Linear(n_dim, n_dim),
            nn.ReLU(),
            nn.Linear(n_dim, n_class),
        ).cuda()
    elif model_type == "prototype":
        model = nn.Linear(n_dim, n_class, bias=False).cuda()
        classmeans = torch.zeros(n_class, n_dim)
        for i in range(n_class):
            if training_modality == "img":
                classmeans[i] = F.normalize(
                    img_features_train[img_labels_train[:, i] == 1].mean(0), dim=0
                )
            elif training_modality == "txt":
                classmeans[i] = F.normalize(
                    txt_features_train[txt_labels_train[:, i] == 1].mean(0), dim=0
                )
        model.weight.data = classmeans.cuda()
    else:
        raise ValueError("Unknown model type")

    img_dataloader_train = DataLoader(
        TensorDataset(img_features_train, img_labels_train),
        batch_size=128,
        shuffle=True,
    )
    txt_dataloader_train = DataLoader(
        TensorDataset(txt_features_train, txt_labels_train),
        batch_size=128,
        shuffle=True,
    )
    img_dataloader_val = DataLoader(
        TensorDataset(img_features_val, img_labels_val), batch_size=128, shuffle=False
    )
    txt_dataloader_val = DataLoader(
        TensorDataset(txt_features_val, txt_labels_val), batch_size=128, shuffle=False
    )

    optimizer = get_optimizer(model, optimizer_type)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    wandb.init(project="crossmodal_transfer")
    for i in trange(n_epochs):
        img_results_train = evaluate(
            img_dataloader_train, model, multilabel=(dataset == "coco")
        )
        img_results_val = evaluate(
            img_dataloader_val, model, multilabel=(dataset == "coco")
        )
        txt_results_train = evaluate(
            txt_dataloader_train, model, multilabel=(dataset == "coco")
        )
        txt_results_val = evaluate(
            txt_dataloader_val, model, multilabel=(dataset == "coco")
        )
        if dataset == "coco":
            img_preds_val = img_results_val["preds"]
            txt_preds_val = txt_results_val["preds"]
            consistency = np.mean((img_preds_val == txt_preds_val).astype(np.float32))
            exact_match_consistency = np.mean(
                (img_preds_val == txt_preds_val).all(1).astype(np.float32)
            )
            wandb.log(
                {
                    "val/consistency": consistency,
                    "val/exact_match_consistency": exact_match_consistency,
                }
            )
        img_results_train.pop("preds")
        img_results_val.pop("preds")
        txt_results_train.pop("preds")
        txt_results_val.pop("preds")
        wandb.log({f"train/img_{k}": v for k, v in img_results_train.items()})
        wandb.log({f"val/img_{k}": v for k, v in img_results_val.items()})
        wandb.log({f"train/txt_{k}": v for k, v in txt_results_train.items()})
        wandb.log({f"val/txt_{k}": v for k, v in txt_results_val.items()})

        if model_type == "prototype":
            break

        if training_modality == "image":
            train_one_epoch(
                img_dataloader_train, model, optimizer, multilabel=(dataset == "coco")
            )
        elif training_modality == "text":
            train_one_epoch(
                txt_dataloader_train, model, optimizer, multilabel=(dataset == "coco")
            )
        else:
            raise ValueError("invalid argument")
        wandb.log({"lr": optimizer.param_groups[0]["lr"]})
        # lr_scheduler.step()

    wandb.finish()

    torch.save(
        model.state_dict(),
        f"{dataset}_{model_type}_{training_modality}_{optimizer_type}_{gap_method}.pt",
    )


if __name__ == "__main__":
    main()
