import json
import os
import random
from collections import defaultdict
from typing import Dict, List

base_dir = "data/COCO"

data: List[Dict] = []

captions_train = json.load(open(f"{base_dir}/annotations/captions_train2017.json"))
captions_val = json.load(open(f"{base_dir}/annotations/captions_val2017.json"))
instances_train = json.load(open(f"{base_dir}/annotations/instances_train2017.json"))
instances_val = json.load(open(f"{base_dir}/annotations/instances_val2017.json"))

assert captions_train["images"] == instances_train["images"]
assert captions_val["images"] == instances_val["images"]
assert instances_train["categories"] == instances_val["categories"]
assert len(instances_train["categories"]) == 80

image2captions = defaultdict(list)
image2categories = defaultdict(list)

cid2id = {item["id"]: idx for idx, item in enumerate(instances_train["categories"])}
id2name = {idx: item["name"] for idx, item in enumerate(instances_train["categories"])}

image2split = defaultdict(str)
for item in captions_train["images"]:
    image2split[item["id"]] = "train"
for item in captions_val["images"]:
    image2split[item["id"]] = "val"

for annotation in captions_train["annotations"] + captions_val["annotations"]:
    image2captions[annotation["image_id"]].append(annotation["caption"])
for key in image2captions:
    image2captions[key] = sorted(list(set(image2captions[key])))

for annotation in instances_train["annotations"] + instances_val["annotations"]:
    image2categories[annotation["image_id"]].append(annotation["category_id"])
for key in image2categories:
    image2categories[key] = sorted(list(set(image2categories[key])))

data = []
random.seed(1234)
for image in captions_train["images"] + captions_val["images"]:
    image_id = image["id"]
    random_caption = random.choice(image2captions[image_id])
    label_ids = [cid2id[cid] for cid in image2categories[image_id]]
    label_names = [id2name[lid] for lid in label_ids]
    label_vec = [0 if lid not in label_ids else 1 for lid in range(80)]
    datasplit = image2split[image_id]
    item = {
        "image": f"{base_dir}/images/{datasplit}2017/{image['file_name']}",
        "text": random_caption,  # TODO: use all captions in the future
        "label": label_vec,
        "attributes": {
            "split": datasplit,
            "label_id": label_ids,
            "label_name": label_names,
        },
    }
    data.append(item)

os.mkdir(f"{base_dir}/processed_attribute_dataset")
with open(f"{base_dir}/processed_attribute_dataset/attributes.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
