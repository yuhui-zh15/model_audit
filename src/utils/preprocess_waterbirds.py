import json
import os

import pandas as pd  # type: ignore

base_dir = "data/Waterbird"
annotations = pd.read_csv(
    f"{base_dir}/raw/waterbird_complete95_forest2water2/metadata.csv"
).to_dict(orient="records")

data = []
for annotation in annotations:
    image_filename = annotation["img_filename"]
    attributes = {
        "species": image_filename.split("/")[0].split(".")[1].replace("_", " "),
        "place_raw": annotation["place_filename"],
        "place": " ".join(annotation["place_filename"].split("/")[2:-1]).replace(
            "_", " "
        ),
        "waterbird": annotation["y"],
        "waterplace": annotation["place"],
    }
    attributes["split"] = ["train", "val", "test"][annotation["split"]]
    data.append(
        {
            "image": f"{base_dir}/processed_attribute_dataset/images/{image_filename}",
            "attributes": attributes,
        }
    )

os.mkdir(f"{base_dir}/processed_attribute_dataset")

with open(f"{base_dir}/processed_attribute_dataset/attributes.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

os.system(
    f"cp -r {base_dir}/raw/waterbird_complete95_forest2water2 \
        {base_dir}/processed_attribute_dataset/images/"
)
