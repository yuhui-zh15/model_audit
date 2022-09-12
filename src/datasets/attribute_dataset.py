import json
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image  # type: ignore
from torch.utils.data import DataLoader, Dataset


class AttributeDataset(Dataset):
    """
    Attribute Dataset.
    The directory structure is assumed to be:
    - root/attributes.jsonl
        A list of dictionaries, each dictionary contains:
        - image: str
        - attributes: dict (special key: split)
    - root/images/
        A directory containing all images.
    """

    def __init__(
        self,
        path: str,
        filter_func: Optional[Callable] = None,
        label_func: Optional[Callable] = None,
        max_data_size: Optional[int] = None,
    ) -> None:
        self.path = path
        self.filter_func = filter_func
        self.label_func = label_func
        self.max_data_size = max_data_size

        attributes = [json.loads(line) for line in open(f"{path}/attributes.jsonl")]

        self.data = []
        for item in attributes:
            if label_func is not None:
                item["label"] = label_func(item)
            if filter_func is None or filter_func(item) is True:
                self.data.append(item)

        if self.max_data_size is not None and len(self.data) > self.max_data_size:
            indices = random.sample(range(len(self.data)), self.max_data_size)
            self.data = [self.data[i] for i in indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, int, Dict]:
        filename, label = self.data[idx]["image"], self.data[idx].get("label", None)
        image = Image.open(f"{self.path}/{filename}")
        return image, label, self.data[idx]


def create_dataloader(
    dataset: AttributeDataset,
    transform: Callable,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    def collate_fn(
        batch: List, transform: Callable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images, labels, _ = zip(*batch)
        image_inputs = torch.stack([transform(image) for image in images], dim=0)
        labels = torch.tensor(labels)
        return image_inputs, labels

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, transform),
    )
    return dataloader


if __name__ == "__main__":
    dataset = AttributeDataset(
        path="/pasteur/u/yuhuiz/data/CelebA/processed_attribute_dataset/",
        filter_func=lambda x: x["attributes"]["split"] == "train",
        label_func=lambda x: int(x["attributes"]["Male"] == 1),
    )
    print(f"{len(dataset)=}")
    print(f"{dataset[0]=}")
    dataLoader = create_dataloader(
        dataset=dataset, transform=lambda x: torch.zeros(100)
    )
