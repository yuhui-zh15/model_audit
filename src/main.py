import itertools
import json

import clip  # type: ignore
import torch

from datasets import AttributeDataset, ImageDataset, TextDataset, create_dataloader
from models import Linear
from trainer import run_one_epoch
from utils import computing_subgroup_metrics, subgrouping

CLIP_MODEL = "ViT-B/32"


def train_waterbird():
    clip_model, transform = clip.load(name=CLIP_MODEL, device="cuda")
    clip_model = clip_model.float()
    model = Linear(clip_model.visual.output_dim, 2).cuda()

    image_dataset_train = AttributeDataset(
        path="/pasteur/u/yuhuiz/data/Waterbird/processed_attribute_dataset/",
        filter_func=lambda x: x["attributes"]["split"] == "train",
        label_func=lambda x: x["attributes"]["waterbird"],
    )
    image_dataloader_train = create_dataloader(
        dataset=image_dataset_train, modality="image", transform=transform, shuffle=True
    )

    image_dataset_val = AttributeDataset(
        path="/pasteur/u/yuhuiz/data/Waterbird/processed_attribute_dataset/",
        filter_func=lambda x: x["attributes"]["split"] == "val",
        label_func=lambda x: x["attributes"]["waterbird"],
    )
    image_dataloader_val = create_dataloader(
        dataset=image_dataset_val, modality="image", transform=transform, shuffle=False
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch_idx in range(10):
        run_one_epoch(
            dataloader=image_dataloader_train,
            model=model,
            clip_model=clip_model,
            modality="image",
            opt=opt,
            epoch_idx=epoch_idx,
            eval=False,
            verbose=True,
        )

        image_metrics_train = run_one_epoch(
            dataloader=image_dataloader_train,
            model=model,
            clip_model=clip_model,
            modality="image",
            opt=None,
            epoch_idx=epoch_idx,
            eval=True,
            verbose=True,
        )

        image_metrics_val = run_one_epoch(
            dataloader=image_dataloader_val,
            model=model,
            clip_model=clip_model,
            modality="image",
            opt=None,
            epoch_idx=epoch_idx,
            eval=True,
            verbose=True,
        )

        print(
            f"Epoch {epoch_idx}: {image_metrics_train['acc']=}, {image_metrics_train['loss']=}, \
            {image_metrics_val['acc']=}, {image_metrics_val['loss']=}"
        )

    torch.save(model.state_dict(), "waterbird_linear_model.pt")


def eval_waterbird():
    clip_model, transform = clip.load(name=CLIP_MODEL, device="cuda")
    clip_model = clip_model.float()
    model = Linear(clip_model.visual.output_dim, 2).cuda()
    state_dict = torch.load("waterbird_linear_model.pt")
    model.load_state_dict(state_dict)

    fields = ["waterbird", "waterplace"]

    image_data = [
        json.loads(line)
        for line in open(
            "/pasteur/u/yuhuiz/data/Waterbird/processed_attribute_dataset/attributes.jsonl"
        )
    ]

    def filter_fn(x):
        return x["attributes"]["split"] == "val"

    image_data = [x for x in image_data if filter_fn(x)]

    def label_fn(x):
        return x["attributes"]["waterbird"]

    for item in image_data:
        item["label"] = label_fn(item)

    image_dataset = ImageDataset(data=image_data)
    image_dataloader = create_dataloader(
        dataset=image_dataset, modality="image", transform=transform
    )
    image_metrics = run_one_epoch(
        dataloader=image_dataloader,
        model=model,
        clip_model=clip_model,
        modality="image",
        opt=None,
        epoch_idx=-1,
        eval=True,
        verbose=True,
    )
    image_preds, image_labels = image_metrics["preds"], image_metrics["labels"]
    image_subgroups = subgrouping(image_data, fields)
    image_subgroup_metrics = computing_subgroup_metrics(
        image_preds, image_labels, image_subgroups
    )
    print(image_subgroup_metrics)

    attributes = {
        "place": set([x["attributes"]["place"] for x in image_data]),
        "species": set([x["attributes"]["species"] for x in image_data]),
    }
    attributes_combinations = [
        dict(zip(attributes, x)) for x in itertools.product(*attributes.values())
    ]
    species_to_label = {
        x["attributes"]["species"]: x["attributes"]["waterbird"] for x in image_data
    }
    places_to_label = {
        x["attributes"]["place"]: x["attributes"]["waterplace"] for x in image_data
    }
    text_data = [
        {
            "text": f"a photo of a {x['species']} in the {x['place']}.",
            "label": species_to_label[x["species"]],
            "attributes": {
                "waterbird": species_to_label[x["species"]],
                "waterplace": places_to_label[x["place"]],
            },
        }
        for x in attributes_combinations
    ]
    text_dataset = TextDataset(data=text_data)
    text_dataloader = create_dataloader(dataset=text_dataset, modality="text")
    text_metrics = run_one_epoch(
        dataloader=text_dataloader,
        model=model,
        clip_model=clip_model,
        modality="text",
        opt=None,
        epoch_idx=-1,
        eval=True,
        verbose=True,
    )
    text_preds, text_labels = text_metrics["preds"], text_metrics["labels"]
    text_subgroups = subgrouping(text_data, fields)
    text_subgroup_metrics = computing_subgroup_metrics(
        text_preds, text_labels, text_subgroups
    )
    print(text_subgroup_metrics)


if __name__ == "__main__":
    train_waterbird()
    eval_waterbird()
