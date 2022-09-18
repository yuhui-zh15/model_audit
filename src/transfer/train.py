import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score  # type: ignore
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange  # type: ignore

import wandb

FEATURE_PATH = "/pasteur/u/yuhuiz/mmdebug/src/pytorch_cache/coco_features_vitb32.pt"
LABEL_PATH = "/pasteur/u/yuhuiz/data/COCO/processed_attribute_dataset/attributes.jsonl"


features = torch.load(FEATURE_PATH)
img_features, txt_features, labels = (
    torch.tensor(features["image_features"]),
    torch.tensor(features["text_features"]),
    torch.tensor(features["labels"]),
)

img_features = F.normalize(img_features)
txt_features = F.normalize(txt_features)

if sys.argv[2] == "centering":
    img_features = img_features - img_features.mean(0)
    txt_features = txt_features - txt_features.mean(0)
    img_features = F.normalize(img_features)
    txt_features = F.normalize(txt_features)
elif sys.argv[2] == "original":
    pass
else:
    raise ValueError("Unknown centering method")

data = [json.loads(line) for line in open(LABEL_PATH)]
train_idxs = [i for i in range(len(data)) if data[i]["attributes"]["split"] == "train"]
val_idxs = [i for i in range(len(data)) if data[i]["attributes"]["split"] == "val"]

img_features_train = img_features[train_idxs]
img_labels_train = labels[train_idxs]
txt_features_train = txt_features[train_idxs]
txt_labels_train = labels[train_idxs]

img_features_val = img_features[val_idxs]
img_labels_val = labels[val_idxs]
txt_features_val = txt_features[val_idxs]
txt_labels_val = labels[val_idxs]

if sys.argv[3] == "no_bias":
    print("No bias")
    linear = nn.Linear(512, 80, bias=False).cuda()
elif sys.argv[3] == "bias":
    print("With bias")
    linear = nn.Linear(512, 80).cuda()
    linear.bias.data.zero_()
else:
    raise ValueError("Unknown bias method")

img_dataloader_train = DataLoader(
    TensorDataset(img_features_train, img_labels_train), batch_size=128, shuffle=True
)
txt_dataloader_train = DataLoader(
    TensorDataset(txt_features_train, txt_labels_train), batch_size=128, shuffle=True
)
img_dataloader_val = DataLoader(
    TensorDataset(img_features_val, img_labels_val), batch_size=128, shuffle=False
)
txt_dataloader_val = DataLoader(
    TensorDataset(txt_features_val, txt_labels_val), batch_size=128, shuffle=False
)


def train_one_epoch(dataloader, model, optimizer, device="cuda"):
    model.train()
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(dataloader, model, device="cuda"):
    model.eval()
    preds = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y.float())

            preds.extend((logits > 0).float().cpu().tolist())
            labels.extend(y.cpu().tolist())
            losses.append(loss.item())
    preds = np.array(preds)
    labels = np.array(labels)
    micro_f1 = f1_score(labels, preds, average="micro")
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")
    samples_f1 = f1_score(labels, preds, average="samples")
    baccs = [balanced_accuracy_score(preds[:, i], labels[:, i]) for i in range(80)]
    bacc = np.mean(baccs)
    loss = np.mean(losses)
    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "samples_f1": samples_f1,
        "bacc": bacc,
        "loss": loss,
    }


if sys.argv[1] == "sgd":
    optimizer = torch.optim.SGD(
        linear.parameters(),
        lr=1e-2,
        momentum=0.9,
        weight_decay=0,
    )
elif sys.argv[1] == "adam":
    optimizer = torch.optim.Adam(  # type: ignore
        linear.parameters(),
        lr=1e-3,
        weight_decay=0,
    )
else:
    raise ValueError("invalid argument")

wandb.init(project="coco_classification_clip_vitb32")
for i in trange(25):
    img_results_train = evaluate(img_dataloader_train, linear)
    img_results_val = evaluate(img_dataloader_val, linear)
    txt_results_train = evaluate(txt_dataloader_train, linear)
    txt_results_val = evaluate(txt_dataloader_val, linear)
    wandb.log({f"train/img_{k}": v for k, v in img_results_train.items()})
    wandb.log({f"val/img_{k}": v for k, v in img_results_val.items()})
    wandb.log({f"train/txt_{k}": v for k, v in txt_results_train.items()})
    wandb.log({f"val/txt_{k}": v for k, v in txt_results_val.items()})

    train_one_epoch(img_dataloader_train, linear, optimizer)

wandb.finish()

torch.save(linear.state_dict(), f"linear_{sys.argv[1:]}.pt")
