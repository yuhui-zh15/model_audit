import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image  # type: ignore
from torch.utils.data import Dataset


class PairedDataset(Dataset):
    """
    Paired Image-Text Dataset.
    The data structure is assumed to be:
    - data: List[Dict]
        A list of dictionaries, each dictionary contains:
        - image: str (path to the image)
        - text: str (caption of the image)
        - label: Optional[List[int]] (list of labels)
        - attributes: Optional[Dict] (including split)
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

    def __getitem__(self, idx: int) -> Tuple[Any, str, Optional[List[int]], Dict]:
        image_file, text, label = (
            self.data[idx]["image"],
            self.data[idx]["text"],
            self.data[idx].get("label", None),
        )
        image = Image.open(image_file)
        return image, text, label, self.data[idx]
