import random
from typing import Dict, List, Optional, Tuple

import clip  # type: ignore
import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """
    Text Dataset.
    The data structure is assumed to be:
    - data: List[Dict]
        A list of dictionaries, each dictionary contains:
        - text: str
        - label: int
    """

    def __init__(
        self,
        data: List[Dict],
        max_data_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.data = data
        self.max_data_size = max_data_size

        if self.max_data_size is not None and len(self.data) > self.max_data_size:
            indices = random.sample(range(len(self.data)), self.max_data_size)
            self.data = [self.data[i] for i in indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, int, Dict]:
        text, label = self.data[idx]["text"], self.data[idx].get("label", None)
        return text, label, self.data[idx]


def create_text_dataloader(
    dataset: TextDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    def collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        texts, labels, _ = zip(*batch)
        text_inputs = clip.tokenize(texts)
        labels = torch.tensor(labels)
        return text_inputs, labels

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return dataloader


if __name__ == "__main__":
    dataset = TextDataset(
        data=[
            {"text": "hello world", "label": 0},
            {"text": "goodbye world", "label": 1},
        ],
    )
    print(f"{len(dataset)=}")
    print(f"{dataset[0]=}")
    dataLoader = create_text_dataloader(dataset=dataset)
