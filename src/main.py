import json

import clip  # type: ignore
import torch

from datasets import AttributeDataset, TextDataset, create_dataloader
from models import Linear
from trainer import run_one_epoch


def main_waterbird():
    clip_model, transform = clip.load(name="ViT-B/32", device="cuda")
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

    attributes = [
        json.loads(line)
        for line in open(
            "/pasteur/u/yuhuiz/data/Waterbird/processed_attribute_dataset/attributes.jsonl"
        )
    ]
    places = set([x["attributes"]["place"] for x in attributes])
    species = set([x["attributes"]["species"] for x in attributes])
    species_to_label = {
        x["attributes"]["species"]: x["attributes"]["waterbird"] for x in attributes
    }

    text_dataset = TextDataset(
        data=[
            {
                "text": f"a photo of a {species} in the {place}.",
                "label": species_to_label[species],
            }
            for species in species
            for place in places
        ],
    )
    text_dataloader = create_dataloader(dataset=text_dataset, modality="text")

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

        text_metrics = run_one_epoch(
            dataloader=text_dataloader,
            model=model,
            clip_model=clip_model,
            modality="text",
            opt=None,
            epoch_idx=epoch_idx,
            eval=True,
            verbose=True,
        )

        print(
            f"Epoch {epoch_idx}: {image_metrics_train['acc']=}, {image_metrics_train['loss']=}, \
            {image_metrics_val['acc']=}, {image_metrics_val['loss']=}, \
            {text_metrics['acc']=}, {text_metrics['loss']=}"
        )

    torch.save(model.state_dict(), "waterbird_linear_model.pt")


def main():
    clip_model, transform = clip.load(name="ViT-B/32", device="cuda")
    clip_model = clip_model.float()
    model = Linear(clip_model.visual.output_dim, 2).cuda()

    dataset = AttributeDataset(
        path="/pasteur/u/yuhuiz/data/CelebA/processed_attribute_dataset/",
        filter_func=lambda x: x["attributes"]["split"] == "val",
        label_func=lambda x: int(x["attributes"]["Male"] == 1),
    )
    dataloader = create_dataloader(
        dataset=dataset, modality="image", transform=transform
    )

    text_dataset = TextDataset(
        data=[
            {"text": "a photo of a man.", "label": 1},
            {"text": "a photo of a woman.", "label": 0},
        ],
    )
    text_dataloader = create_dataloader(dataset=text_dataset, modality="text")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch_idx in range(10):
        run_one_epoch(
            dataloader=dataloader,
            model=model,
            clip_model=clip_model,
            modality="image",
            opt=opt,
            epoch_idx=epoch_idx,
            eval=False,
            verbose=True,
        )

        metrics = run_one_epoch(
            dataloader=dataloader,
            model=model,
            clip_model=clip_model,
            modality="image",
            opt=None,
            epoch_idx=epoch_idx,
            eval=True,
            verbose=True,
        )

        print(f"{metrics['acc']=}, {metrics['loss']=}")

        text_metrics = run_one_epoch(
            dataloader=text_dataloader,
            model=model,
            clip_model=clip_model,
            modality="text",
            opt=None,
            epoch_idx=epoch_idx,
            eval=True,
            verbose=True,
        )

        print(f"{text_metrics['acc']=}, {text_metrics['loss']=}")


if __name__ == "__main__":
    # main()
    main_waterbird()
