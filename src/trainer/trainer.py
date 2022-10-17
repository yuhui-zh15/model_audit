from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from clip.model import CLIP  # type: ignore
from sklearn.metrics import balanced_accuracy_score  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore


def run_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    clip_model: CLIP,
    modality: str,
    opt: Optional[optim.Optimizer] = None,
    epoch_idx: int = -1,
    eval: bool = False,
    verbose: bool = False,
    multilabel: bool = False,
    normalize: bool = True,
) -> Dict:

    if not eval:
        assert opt is not None

    model = model.train() if not eval else model.eval()
    clip_model = clip_model.eval()

    losses, preds, labels, logits, features = [], [], [], [], []
    bar = (
        tqdm(dataloader, desc=f"Epoch {epoch_idx}, Eval {eval}")
        if verbose
        else dataloader
    )
    for batch_idx, batch in enumerate(bar):
        x, y = batch
        x, y = x.cuda(), y.cuda()

        with torch.no_grad():
            if modality == "image":
                h = clip_model.encode_image(x)
            elif modality == "text":
                h = clip_model.encode_text(x)
            else:
                raise ValueError(f"Invalid modality: {modality}")

        if normalize:
            h = F.normalize(h)

        logits_ = model(h)

        if multilabel:
            loss = F.binary_cross_entropy_with_logits(logits_, y.float())
        else:
            loss = F.cross_entropy(logits_, y)

        if not eval:
            opt.zero_grad()  # type: ignore
            loss.backward()
            opt.step()  # type: ignore

        if multilabel:
            preds.extend((logits_ > 0).float().detach().cpu().tolist())
        else:
            preds.extend(logits_.argmax(dim=1).detach().cpu().tolist())
        losses.append(loss.item())
        labels.extend(y.detach().cpu().tolist())
        logits.extend(logits_.detach().cpu().tolist())
        features.extend(h.detach().cpu().tolist())

    if multilabel:
        accs = [
            balanced_accuracy_score(np.array(labels)[:, i], np.array(preds)[:, i])
            for i in range(np.array(preds).shape[1])
        ]
        acc = sum(accs) / len(accs)
    else:
        acc = np.mean(np.array(preds) == np.array(labels))
    mean_loss = np.mean(losses)
    return {
        "loss": mean_loss,
        "acc": acc,
        "preds": preds,
        "labels": labels,
        "logits": logits,
        "features": features,
    }


def extract_features(
    dataloader: DataLoader, clip_model: CLIP, modality: str, verbose: bool = False
) -> torch.Tensor:

    clip_model = clip_model.eval()

    features = []
    bar = (
        tqdm(dataloader, desc=f"Extracting features for {modality}")
        if verbose
        else dataloader
    )
    for batch_idx, batch in enumerate(bar):
        x, _ = batch
        x = x.cuda()

        with torch.no_grad():
            if modality == "image":
                h = clip_model.encode_image(x)
            elif modality == "text":
                h = clip_model.encode_text(x)
            else:
                raise ValueError(f"Invalid modality: {modality}")

        features.extend(h.detach().cpu().tolist())

    features_tensor = torch.tensor(features)
    return features_tensor
