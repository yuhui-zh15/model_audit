from typing import Callable, List, Optional, Tuple

import clip  # type: ignore
import torch
from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    dataset: Dataset,
    modality: str,
    transform: Optional[Callable] = None,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    def collate_fn(batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_inputs, labels, _ = zip(*batch)
        if modality == "image":
            assert (
                transform is not None
            ), "transform must be provided for image modality"
            inputs = torch.stack([transform(image) for image in raw_inputs], dim=0)
            _ = [image.close() for image in raw_inputs]
        elif modality == "text":
            inputs = clip.tokenize(raw_inputs)
        else:
            raise ValueError(f"Unknown modality: {modality}")
        labels = torch.tensor(labels)
        return inputs, labels

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return dataloader
