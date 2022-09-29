import json
import os

import pandas as pd  # type: ignore

base_dir = "data/FairFace"
annotations_train = pd.read_csv(f"{base_dir}/raw/fairface_label_train.csv").to_dict(
    orient="records"
)
annotations_val = pd.read_csv(f"{base_dir}/raw/fairface_label_val.csv").to_dict(
    orient="records"
)
annotations = annotations_train + annotations_val

data = []
for idx, annotation in enumerate(annotations):
    filename = annotation["file"]
    attributes = {key: value for key, value in annotation.items() if key != "file"}
    attributes["split"] = ["train", "val"][int(idx >= len(annotations_train))]
    data.append(
        {
            "image": f"{base_dir}/processed_attribute_dataset/images/{filename}",
            "attributes": attributes,
        }
    )

os.mkdir(f"{base_dir}/processed_attribute_dataset")

with open(f"{base_dir}/processed_attribute_dataset/attributes.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

os.mkdir(f"{base_dir}/processed_attribute_dataset/images")

os.system(
    f"cp -r {base_dir}/raw/train \
        {base_dir}/processed_attribute_dataset/images/"
)
os.system(
    f"cp -r {base_dir}/raw/val \
        {base_dir}/processed_attribute_dataset/images/"
)
