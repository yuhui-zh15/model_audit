import json
import os

import pandas as pd  # type: ignore

base_dir = "/sailhome/yuhuiz/data/CelebA"
annotations = pd.read_csv(f"{base_dir}/raw/list_attr_celeba.csv").to_dict(
    orient="records"
)
partitions = pd.read_csv(f"{base_dir}/raw/list_eval_partition.csv").to_dict(
    orient="records"
)
assert len(annotations) == len(partitions)

data = []
for partition, annotation in zip(partitions, annotations):
    assert partition["image_id"] == annotation["image_id"]
    image_id = annotation["image_id"]
    attributes = {
        key: annotation[key] for key in annotation.keys() if key != "image_id"
    }
    partition = ["train", "val", "test"][partition["partition"]]
    assert "split" not in attributes
    attributes["split"] = partition
    data.append(
        {
            "image": f"{base_dir}/processed_attribute_dataset/images/{image_id}",
            "attributes": attributes,
        }
    )

os.mkdir(f"{base_dir}/processed_attribute_dataset")

with open(f"{base_dir}/processed_attribute_dataset/attributes.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

os.system(
    f"cp -r {base_dir}/raw/img_align_celeba/img_align_celeba/ \
        {base_dir}/processed_attribute_dataset/images/"
)
