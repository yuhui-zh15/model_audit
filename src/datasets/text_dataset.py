import random
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset


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
            random.seed(1234)
            indices = random.sample(range(len(self.data)), self.max_data_size)
            self.data = [self.data[i] for i in indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, int, Dict]:
        text, label = self.data[idx]["text"], self.data[idx].get("label", None)
        return text, label, self.data[idx]
