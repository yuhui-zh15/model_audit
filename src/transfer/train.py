import json
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange  # type: ignore

import wandb

BASE_PATH = "/sailhome/yuhuiz/develop/multimodal_debugging/"


def split_features_unbalanced(features: torch.Tensor, labels: torch.Tensor):
    random.seed(1234)
    N, D = features.shape
    N_train = int(N * 0.8)
    train_idxs = sorted(random.sample(range(N), N_train))
    val_idxs = [i for i in range(N) if i not in train_idxs]
    train_features = features[train_idxs, :]
    val_features = features[val_idxs, :]
    train_labels = labels[train_idxs]
    val_labels = labels[val_idxs]
    return train_features, train_labels, val_features, val_labels


img_features = F.normalize(
    torch.load(
        f"{BASE_PATH}/archive/trained_models/img_features_imagenet_clip_vitb32.pt"
    ).cuda(),
    p=2,
    dim=1,
)
img_labels = torch.tensor([i for i in range(1000) for _ in range(50)]).cuda()
text_features = F.normalize(
    torch.load(
        f"{BASE_PATH}/archive/trained_models/text_features_imagenet_clip_vitb32.pt"
    ).cuda(),
    p=2,
    dim=1,
)
text_labels = torch.tensor([i for i in range(1000) for _ in range(80)]).cuda()

(
    img_features_train,
    img_labels_train,
    img_features_val,
    img_labels_val,
) = split_features_unbalanced(img_features, img_labels)
(
    text_features_train,
    text_labels_train,
    text_features_val,
    text_labels_val,
) = split_features_unbalanced(text_features, text_labels)

if sys.argv[2] == "centering":
    img_features -= img_features.mean(0)
    img_features = F.normalize(img_features)
    text_features -= text_features.mean(0)
    text_features = F.normalize(text_features)
elif sys.argv[2] == "original":
    pass
else:
    raise ValueError("Unknown centering method")

if sys.argv[3] == "no_bias":
    print("No bias")
    linear = nn.Linear(512, 1000, bias=False).cuda()
elif sys.argv[3] == "bias":
    print("With bias")
    linear = nn.Linear(512, 1000).cuda()
    linear.bias.data.zero_()
else:
    raise ValueError("Unknown bias method")

img_dataloader_train = DataLoader(
    TensorDataset(img_features_train, img_labels_train), batch_size=128, shuffle=True
)
img_dataloader_val = DataLoader(
    TensorDataset(img_features_val, img_labels_val), batch_size=128, shuffle=True
)
text_dataloader_train = DataLoader(
    TensorDataset(text_features_train, text_labels_train), batch_size=128, shuffle=True
)
text_dataloader_val = DataLoader(
    TensorDataset(text_features_val, text_labels_val), batch_size=128, shuffle=True
)

LOGIT_CONSTANT = 10


def train_one_epoch(dataloader, model, optimizer, device="cuda"):
    model.train()
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        pred = model(x) * LOGIT_CONSTANT
        loss = F.cross_entropy(pred, y)
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
            pred = model(x) * LOGIT_CONSTANT
            loss = F.cross_entropy(pred, y)
            preds.append(pred.argmax(1))
            labels.append(y)
            losses.append(loss.item())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    acc = (preds == labels).float().mean().item()
    loss = np.mean(losses)
    return acc, loss, preds, labels


weight_decay = 1e-3
lr = 1e-2
momentum = 0.9
if sys.argv[1] == "wd_all":
    print("Weight decay")
    optimizer = torch.optim.SGD(
        linear.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
elif sys.argv[1] == "wd_weight":
    print("No weight decay")
    optimizer = torch.optim.SGD(
        [
            {"params": linear.weight, "weight_decay": weight_decay},
            {"params": linear.bias, "weight_decay": 0},
        ],
        lr=lr,
        momentum=momentum,
    )
elif sys.argv[1] == "no_wd":
    print("No weight decay all")
    optimizer = torch.optim.SGD(
        linear.parameters(),
        lr=lr,
        momentum=momentum,
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

wandb.init(project="imagenet_classification_clip_vitb32")
logs = []
for i in trange(100):
    img_acc_train, img_loss_train, _, _ = evaluate(img_dataloader_train, linear)
    img_acc_val, img_loss_val, _, _ = evaluate(img_dataloader_val, linear)
    text_acc_train, text_loss_train, _, _ = evaluate(text_dataloader_train, linear)
    text_acc_val, text_loss_val, _, _ = evaluate(text_dataloader_val, linear)
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
    open(f"{BASE_PATH}/training_logs_1layer_img_{sys.argv[1]}.json", "w"),
    indent=2,
)
wandb.finish()

torch.save(linear.state_dict(), f"{BASE_PATH}/linear_{sys.argv[1]}.pt")
