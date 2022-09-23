import json
from typing import Union

import clip  # type: ignore
import torch

import wandb
from datasets import ImageDataset, TextDataset, create_dataloader
from models import Linear
from trainer import run_one_epoch

CLIP_MODEL = "ViT-B/32"
COCO_PATH = "/pasteur/u/yuhuiz/data/COCO/processed_attribute_dataset"
COCO_NUM_CLS = 80
N_EPOCHS = 50


def train_coco(train_modality: str) -> None:
    wandb.init(project="mmdebug")

    clip_model, transform = clip.load(name=CLIP_MODEL, device="cuda")
    clip_model = clip_model.float()
    model = Linear(clip_model.visual.output_dim, COCO_NUM_CLS).cuda()

    data = [json.loads(line) for line in open(f"{COCO_PATH}/attributes.jsonl")]
    data_train = [x for x in data if x["attributes"]["split"] == "train"]
    data_val = [x for x in data if x["attributes"]["split"] == "val"]

    dataset_train: Union[ImageDataset, TextDataset]
    if train_modality == "image":
        dataset_train = ImageDataset(data_train)
        dataloader_train = create_dataloader(
            dataset=dataset_train,
            modality="image",
            transform=transform,
            shuffle=True,
            batch_size=64,
            num_workers=8,
        )
    elif train_modality == "text":
        dataset_train = TextDataset(data_train)
        dataloader_train = create_dataloader(
            dataset=dataset_train,
            modality="text",
            shuffle=True,
            batch_size=64,
            num_workers=8,
        )
    else:
        raise ValueError(f"Unknown train_modality: {train_modality}")

    image_dataset_val = ImageDataset(data_val)
    image_dataloader_val = create_dataloader(
        dataset=image_dataset_val, modality="image", transform=transform, shuffle=False
    )

    text_dataset_val = TextDataset(data_val)
    text_dataloader_val = create_dataloader(
        dataset=text_dataset_val, modality="text", shuffle=False
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch_idx in range(N_EPOCHS):
        image_metrics_train = run_one_epoch(
            dataloader=dataloader_train,
            model=model,
            clip_model=clip_model,
            modality=train_modality,
            opt=opt,
            epoch_idx=epoch_idx,
            eval=False,
            verbose=True,
            multilabel=True,
        )

        image_metrics_val = run_one_epoch(
            dataloader=image_dataloader_val,
            model=model,
            clip_model=clip_model,
            modality="image",
            opt=None,
            epoch_idx=epoch_idx,
            eval=True,
            verbose=True,
            multilabel=True,
        )

        text_metrics_val = run_one_epoch(
            dataloader=text_dataloader_val,
            model=model,
            clip_model=clip_model,
            modality="text",
            opt=None,
            epoch_idx=epoch_idx,
            eval=True,
            verbose=True,
            multilabel=True,
        )

        wandb.log(
            {
                "train/loss": image_metrics_train["loss"],
                "train/acc": image_metrics_train["acc"],
                "val/img_loss": image_metrics_val["loss"],
                "val/img_acc": image_metrics_val["acc"],
                "val/txt_loss": text_metrics_val["loss"],
                "val/txt_acc": text_metrics_val["acc"],
            }
        )

    torch.save(model.state_dict(), "coco_linear_model.pt")


def extract_features():
    clip_model, transform = clip.load(name=CLIP_MODEL, device="cuda")
    clip_model = clip_model.float()
    model = Linear(clip_model.visual.output_dim, COCO_NUM_CLS).cuda()

    data = [json.loads(line) for line in open(f"{COCO_PATH}/attributes.jsonl")]

    image_dataset = ImageDataset(data)
    image_dataloader = create_dataloader(
        dataset=image_dataset, modality="image", transform=transform, shuffle=False
    )

    text_dataset = TextDataset(data)
    text_dataloader = create_dataloader(
        dataset=text_dataset, modality="text", shuffle=False
    )

    image_metrics = run_one_epoch(
        dataloader=image_dataloader,
        model=model,
        clip_model=clip_model,
        modality="image",
        opt=None,
        epoch_idx=-1,
        eval=True,
        verbose=True,
        multilabel=True,
        normalize=False,
    )

    text_metrics = run_one_epoch(
        dataloader=text_dataloader,
        model=model,
        clip_model=clip_model,
        modality="text",
        opt=None,
        epoch_idx=-1,
        eval=True,
        verbose=True,
        multilabel=True,
        normalize=False,
    )

    assert image_metrics["labels"] == text_metrics["labels"]

    torch.save(
        {
            "image_features": image_metrics["features"],
            "text_features": text_metrics["features"],
            "labels": image_metrics["labels"],
        },
        "coco_features_vitb32.pt",
    )


if __name__ == "__main__":
    # train_coco(sys.argv[1])
    extract_features()
