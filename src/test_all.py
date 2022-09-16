import json

import torch

from datasets import (
    AttributeDataset,
    ImageDataset,
    PairedDataset,
    TextDataset,
    create_dataloader,
)

CELEBA_PATH = "/pasteur/u/yuhuiz/data/CelebA/processed_attribute_dataset"
WATERBIRDS_PATH = "/pasteur/u/yuhuiz/data/Waterbird/processed_attribute_dataset"
FAIRFACE_PATH = "/pasteur/u/yuhuiz/data/FairFace/processed_attribute_dataset"
TRIANGLESQUARE_PATH = (
    "/pasteur/u/yuhuiz/data/TriangleSquare/processed_attribute_dataset"
)
COCO_PATH = "/pasteur/u/yuhuiz/data/COCO/processed_attribute_dataset"


def test_image_dataset():
    dataset = ImageDataset(
        data=[
            {"image": f"{CELEBA_PATH}/images/056531.jpg", "label": 0},
            {"image": f"{CELEBA_PATH}/images/079043.jpg", "label": 1},
        ],
    )
    assert len(dataset) == 2
    dataloader = create_dataloader(
        dataset=dataset, modality="image", transform=lambda x: torch.zeros(100)
    )
    assert dataloader is not None


def test_attribute_dataset_waterbirds():
    dataset = AttributeDataset(
        path=WATERBIRDS_PATH,
        filter_func=lambda x: x["attributes"]["split"] == "val",
        label_func=lambda x: x["attributes"]["waterbird"],
    )
    assert len(dataset) == 1199
    dataloader = create_dataloader(
        dataset=dataset, modality="image", transform=lambda x: torch.zeros(100)
    )
    assert dataloader is not None


def test_attribute_dataset_celeba():
    dataset = AttributeDataset(
        path=CELEBA_PATH,
        filter_func=lambda x: x["attributes"]["split"] == "val",
        label_func=lambda x: int(x["attributes"]["Male"] == 1),
    )
    assert len(dataset) == 19867
    dataloader = create_dataloader(
        dataset=dataset, modality="image", transform=lambda x: torch.zeros(100)
    )
    assert dataloader is not None


def test_attribute_dataset_fairface():
    dataset = AttributeDataset(
        path=FAIRFACE_PATH,
        filter_func=lambda x: x["attributes"]["split"] == "val",
        label_func=lambda x: int(x["attributes"]["gender"] == "Male"),
    )
    assert len(dataset) == 10954
    dataloader = create_dataloader(
        dataset=dataset, modality="image", transform=lambda x: torch.zeros(100)
    )
    assert dataloader is not None


def test_attribute_dataset_trianglesquare():
    dataset = AttributeDataset(
        path=TRIANGLESQUARE_PATH,
        filter_func=None,
        label_func=lambda x: x["attributes"]["label"],
    )
    assert len(dataset) == 10000
    dataloader = create_dataloader(
        dataset=dataset, modality="image", transform=lambda x: torch.zeros(100)
    )
    assert dataloader is not None


def test_text_dataset():
    dataset = TextDataset(
        data=[
            {"text": "hello world", "label": 0},
            {"text": "goodbye world", "label": 1},
        ],
    )
    assert len(dataset) == 2
    dataloader = create_dataloader(dataset=dataset, modality="text")
    assert dataloader is not None


def test_paired_dataset_coco():
    def filter_func(x):
        return x["attributes"]["split"] == "val"

    data = [json.loads(line) for line in open(f"{COCO_PATH}/attributes.jsonl")]
    filtered_data = [x for x in data if filter_func(x)]
    dataset = PairedDataset(data=filtered_data)
    assert len(dataset) == 5000
    dataloader = create_dataloader(
        dataset=dataset,
        modality="image",
        transform=lambda x: torch.zeros(100),
        paired=True,
    )
    assert dataloader is not None


if __name__ == "__main__":
    test_image_dataset()
    test_attribute_dataset_celeba()
    test_attribute_dataset_waterbirds()
    test_attribute_dataset_fairface()
    test_attribute_dataset_trianglesquare()
    test_text_dataset()
    test_paired_dataset_coco()
