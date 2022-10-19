import json
from typing import Callable, List

import clip  # type: ignore
import numpy as np
import torch

from datasets import ImageDataset, TextDataset, create_dataloader
from models import Linear
from trainer import run_one_epoch
from utils import computing_subgroup_metrics, subgrouping


def rectify_models(
    clip_model_name: str,
    linear_model_path: str,
    data_path: str,
    filter_fn: Callable,
    label_fn: Callable,
    prepare_fn: Callable,
    fields: List[str],
) -> None:
    clip_model, transform = clip.load(name=clip_model_name, device="cuda")
    clip_model = clip_model.float()
    state_dict = torch.load(linear_model_path)
    n_class = state_dict["fc.weight"].shape[0]
    model = Linear(clip_model.visual.output_dim, n_class).cuda()
    model.load_state_dict(state_dict)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    text_data = prepare_fn(data_path, "ensemble")
    text_dataset = TextDataset(data=text_data)
    text_dataloader = create_dataloader(
        dataset=text_dataset, modality="text", shuffle=True
    )

    image_data = [json.loads(line) for line in open(data_path)]
    image_data = [item for idx, item in enumerate(image_data) if filter_fn(idx, item)]
    for item in image_data:
        item["label"] = label_fn(item)
    image_dataset = ImageDataset(data=image_data)
    image_dataloader = create_dataloader(
        dataset=image_dataset, modality="image", transform=transform
    )

    for epoch in range(10):

        image_metrics = run_one_epoch(
            dataloader=image_dataloader,
            model=model,
            clip_model=clip_model,
            modality="image",
            opt=None,
            epoch_idx=epoch,
            eval=True,
            verbose=True,
        )

        text_metrics = run_one_epoch(  # noqa
            dataloader=text_dataloader,
            model=model,
            clip_model=clip_model,
            modality="text",
            opt=opt,
            epoch_idx=epoch,
            eval=False,
            verbose=True,
        )

        image_preds, image_labels = image_metrics["preds"], image_metrics["labels"]
        image_subgroups = subgrouping(image_data, fields)
        image_instance_accs = np.array(image_preds) == np.array(image_labels)
        image_subgroup_metrics = computing_subgroup_metrics(
            image_instance_accs, image_subgroups
        )

        print(f"Epoch {epoch}")
        print(
            "Subgroup accuracy",
            sorted(image_subgroup_metrics.items(), key=lambda x: x[1]),
        )
        print("Mean accuracy", np.mean(list(image_subgroup_metrics.values())))
        print()
