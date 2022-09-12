import torch

from datasets import AttributeDataset, ImageDataset, TextDataset, create_dataloader

CELEBA_PATH = "/pasteur/u/yuhuiz/data/CelebA/processed_attribute_dataset"


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


def test_attribute_dataset():
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


if __name__ == "__main__":
    test_image_dataset()
    test_attribute_dataset()
    test_text_dataset()
