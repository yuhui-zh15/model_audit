import json
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image  # type: ignore
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Image Dataset.
    The data structure is assumed to be:
    - data: List[Dict]
        A list of dictionaries, each dictionary contains:
        - image: str (path to the image)
        - label: Optional[int]
        - attributes: Optional[Dict]
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

    def __getitem__(self, idx: int) -> Tuple[Any, int, Dict]:
        image_file, label = self.data[idx]["image"], self.data[idx].get("label", None)
        image = Image.open(image_file)
        return image, label, self.data[idx]


class AttributeDataset(ImageDataset):
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

        data = []
        for item in attributes:
            if label_func is not None:
                item["label"] = label_func(item)
            if filter_func is None or filter_func(item) is True:
                data.append(item)

        super().__init__(data)
