from typing import Dict, Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore


def run_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    /,
    *,
    opt: Optional[optim.Optimizer] = None,
    epoch_idx: int = -1,
    eval: bool = False,
    verbose: bool = False,
) -> Dict:

    if not eval:
        assert opt is not None

    model = model.train() if not eval else model.eval()

    losses, preds, labels = [], [], []
    bar = (
        tqdm(dataloader, desc=f"Epoch {epoch_idx}, Eval {eval}")
        if verbose
        else dataloader
    )
    for batch_idx, batch in enumerate(bar):
        x, y = batch
        x, y = x.cuda(), y.cuda()

        logits = model(x)

        loss = F.cross_entropy(logits, y)
        if not eval:
            opt.zero_grad()  # type: ignore
            loss.backward()
            opt.step()  # type: ignore

        losses.append(loss.item())
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())

    acc = np.mean(np.array(preds) == np.array(labels))
    loss = np.mean(losses)
    return {
        "loss": loss,
        "acc": acc,
        "preds": preds,
        "labels": labels,
    }
