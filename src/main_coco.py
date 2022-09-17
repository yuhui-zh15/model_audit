import json
import os
from pprint import pprint

import clip  # type: ignore
import torch

import wandb
from datasets import AttributeDataset, ImageDataset, TextDataset, create_dataloader
from models import Linear
from trainer import run_one_epoch

CLIP_MODEL = "ViT-B/32"
COCO_PATH = "/pasteur/u/yuhuiz/data/COCO/processed_attribute_dataset"
COCO_NUM_CLS = 80


def train_coco():
    wandb.init(project="mmdebug")

    clip_model, transform = clip.load(name=CLIP_MODEL, device="cuda")
    clip_model = clip_model.float()
    model = Linear(clip_model.visual.output_dim, COCO_NUM_CLS).cuda()

    image_dataset_train = AttributeDataset(
        path=COCO_PATH,
        filter_func=lambda x: x["attributes"]["split"] == "train",
        label_func=lambda x: x["label"],
    )
    image_dataloader_train = create_dataloader(
        dataset=image_dataset_train, modality="image", transform=transform, shuffle=True
    )

    image_dataset_val = AttributeDataset(
        path=COCO_PATH,
        filter_func=lambda x: x["attributes"]["split"] == "val",
        label_func=lambda x: x["label"],
    )
    image_dataloader_val = create_dataloader(
        dataset=image_dataset_val, modality="image", transform=transform, shuffle=False
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch_idx in range(10):
        run_one_epoch(
            dataloader=image_dataloader_train,
            model=model,
            clip_model=clip_model,
            modality="image",
            opt=opt,
            epoch_idx=epoch_idx,
            eval=False,
            verbose=True,
            multilabel=True,
        )

        image_metrics_train = run_one_epoch(
            dataloader=image_dataloader_train,
            model=model,
            clip_model=clip_model,
            modality="image",
            opt=None,
            epoch_idx=epoch_idx,
            eval=True,
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
        wandb.log({"train_loss": image_metrics_train["loss"]})
        wandb.log({"train_acc": image_metrics_train["acc"]})
        wandb.log({"val_loss": image_metrics_val["loss"]})
        wandb.log({"val_acc": image_metrics_val["acc"]})

        print(
            f"Epoch {epoch_idx}: {image_metrics_train['acc']=}, {image_metrics_train['loss']=}, \
            {image_metrics_val['acc']=}, {image_metrics_val['loss']=}"
        )

    torch.save(model.state_dict(), "coco_linear_model.pt")


def eval_coco():
    clip_model, transform = clip.load(name=CLIP_MODEL, device="cuda")
    clip_model = clip_model.float()
    model = Linear(clip_model.visual.output_dim, COCO_NUM_CLS).cuda()
    state_dict = torch.load("coco_linear_model.pt")
    model.load_state_dict(state_dict)

    image_data = [
        json.loads(line) for line in open(os.path.join(COCO_PATH, "attributes.jsonl"))
    ]

    def filter_fn(x):
        return x["attributes"]["split"] == "val"

    image_data = [x for x in image_data if filter_fn(x)]

    def label_fn(x):
        return x["label"]

    for item in image_data:
        item["label"] = label_fn(item)

    image_dataset = ImageDataset(data=image_data)
    image_dataloader = create_dataloader(
        dataset=image_dataset, modality="image", transform=transform
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
    )
    pprint(image_metrics["acc"])

    text_data = [
        {
            "text": x["text"],
            "label": x["label"],
        }
        for x in image_data
    ]
    text_dataset = TextDataset(data=text_data)
    text_dataloader = create_dataloader(dataset=text_dataset, modality="text")
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
    )
    pprint(text_metrics["acc"])


if __name__ == "__main__":
    train_coco()
    eval_coco()
