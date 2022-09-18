import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score  # type: ignore
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
    baccs = [balanced_accuracy_score(preds[:, i], labels[:, i]) for i in range(80)]
    bacc = np.mean(baccs)
    loss = np.mean(losses)
    return bacc, loss, preds, labels


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
logs = []
for i in trange(100):
    img_acc_train, img_loss_train, _, _ = evaluate(img_dataloader_train, linear)
    text_acc_train, text_loss_train, _, _ = evaluate(txt_dataloader_train, linear)
    img_acc_val, img_loss_val, _, _ = evaluate(img_dataloader_val, linear)
    text_acc_val, text_loss_val, _, _ = evaluate(txt_dataloader_val, linear)
    logs.append(
        [
            (img_acc_train, img_acc_val, text_acc_train, text_acc_val),
            (img_loss_train, img_loss_val, text_loss_train, text_loss_val),
        ]
    )
    wandb.log(
        {
            "train/img_acc": img_acc_train,
            "val/img_acc": img_acc_val,
            "train/text_acc": text_acc_train,
            "val/text_acc": text_acc_val,
            "train/img_loss": img_loss_train,
            "val/img_loss": img_loss_val,
            "train/text_loss": text_loss_train,
            "val/text_loss": text_loss_val,
        }
    )

    train_one_epoch(img_dataloader_train, linear, optimizer)


json.dump(
    logs,
    open(f"training_logs_1layer_img_{sys.argv[1:]}.json", "w"),
    indent=2,
)
wandb.finish()

torch.save(linear.state_dict(), f"linear_{sys.argv[1:]}.pt")
