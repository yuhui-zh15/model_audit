import json

import clip  # type: ignore
import torch

from datasets import ImageDataset, TextDataset, create_dataloader
from trainer import extract_features


def filter_name(name: str) -> str:
    return "".join([c for c in name.lower() if c.isalnum()])


def extract_features_coco(model_name: str):
    clip_model, transform = clip.load(name=model_name, device="cuda")
    clip_model = clip_model.float()

    data = [
        json.loads(line)
        for line in open("../data/COCO/processed_attribute_dataset/attributes.jsonl")
    ]

    image_dataset = ImageDataset(data)
    image_dataloader = create_dataloader(
        dataset=image_dataset,
        modality="image",
        transform=transform,
        shuffle=False,
        batch_size=1024,
        num_workers=16,
    )
    image_features = extract_features(
        dataloader=image_dataloader,
        clip_model=clip_model,
        modality="image",
        verbose=True,
    )

    text_dataset = TextDataset(data)
    text_dataloader = create_dataloader(
        dataset=text_dataset,
        modality="text",
        shuffle=False,
        batch_size=1024,
        num_workers=16,
    )
    text_features = extract_features(
        dataloader=text_dataloader, clip_model=clip_model, modality="text", verbose=True
    )

    labels = torch.tensor([item["label"] for item in data])

    torch.save(
        {
            "image_features": image_features,
            "text_features": text_features,
            "labels": labels,
        },
        f"coco_features_{filter_name(model_name)}.pt",
    )


def extract_features_imagenet(model_name: str):
    raise NotImplementedError


def extract_features_others(model_name: str, dataset: str):
    clip_model, transform = clip.load(name=model_name, device="cuda")
    clip_model = clip_model.float()

    data = [
        json.loads(line)
        for line in open(
            f"../data/{dataset}/processed_attribute_dataset/attributes.jsonl"
        )
    ]
    for item in data:
        item["label"] = 0

    image_dataset = ImageDataset(data)
    image_dataloader = create_dataloader(
        dataset=image_dataset,
        modality="image",
        transform=transform,
        shuffle=False,
        batch_size=1024,
        num_workers=16,
    )
    image_features = extract_features(
        dataloader=image_dataloader,
        clip_model=clip_model,
        modality="image",
        verbose=True,
    )

    torch.save(
        image_features,
        f"{dataset.lower()}_features_{filter_name(model_name)}.pt",
    )


if __name__ == "__main__":
    extract_features_coco(model_name="ViT-B/32")
    extract_features_imagenet(model_name="ViT-B/32")
    extract_features_others(model_name="ViT-B/32", dataset="Waterbird")
    extract_features_others(model_name="ViT-B/32", dataset="FairFace")
    extract_features_others(model_name="ViT-B/32", dataset="TriangleSquare")
