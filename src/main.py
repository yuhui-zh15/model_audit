import itertools
import json
import random

import clip  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from datasets import AttributeDataset, ImageDataset, TextDataset, create_dataloader
from dro_loss import LossComputer
from models import Linear
from trainer import run_one_epoch
from utils import computing_subgroup_metrics, subgrouping

CLIP_MODEL = "ViT-B/32"


def train_one_epoch(dataloader, model, optimizer, device="cuda"):
    model.train()
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_one_epoch_dro(dataloader, model, optimizer, loss_fn, device="cuda"):
    model.train()
    for batch in dataloader:
        x, y, gidxs = batch
        x, y, gidxs = x.to(device), y.to(device), gidxs.to(device)
        logits = model(x)
        loss = loss_fn.loss(logits, y, group_idx=gidxs, is_training=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(dataloader, model, device="cuda"):
    model.eval()
    preds = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(y.cpu().tolist())
            losses.append(loss.item())
    preds = np.array(preds)
    labels = np.array(labels)
    acc = (preds == labels).mean()
    loss = np.mean(losses)
    return {
        "acc": acc,
        "loss": loss,
        "preds": preds,
        "labels": labels,
    }


def extract_image_features(path: str):
    clip_model, transform = clip.load(name=CLIP_MODEL, device="cuda")
    clip_model = clip_model.float()
    model = Linear(clip_model.visual.output_dim, 10000).cuda()

    data = [json.loads(line) for line in open(path)]
    for item in data:
        item["label"] = 0

    image_dataset = ImageDataset(data)
    image_dataloader = create_dataloader(
        dataset=image_dataset, modality="image", transform=transform, shuffle=False
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
        normalize=False,
    )

    torch.save(image_metrics["features"], f"{path.split('/')[-3]}_features_vitb32.pt")


def train_image_model_fairface(data_path: str, feature_path: str, close_gap: bool):
    data = [json.loads(line) for line in open(data_path)]
    labels = torch.tensor(
        [1 if item["attributes"]["gender"] == "Female" else 0 for item in data]
    )

    train_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "train"
    ]
    val_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "val"
    ]

    montana_race = {
        "White": 90.6 / 100,
        "Indian": 6.2 / 100,
        "East Asian": 0.5 / 100,
        "Black": 0.3 / 100,
    }
    montana_race_norm = {
        key: montana_race[key] / montana_race["White"] for key in montana_race
    }
    print(montana_race_norm)

    sampled_train_idxs = []
    random.seed(1234)
    for idx in train_idxs:
        race = data[idx]["attributes"]["race"]
        if race in montana_race_norm:
            if random.random() <= montana_race_norm[race]:
                sampled_train_idxs.append(idx)

    # race_distribution = Counter([item["attributes"]["race"] for item in sampled_train_data]).most_common()

    features = F.normalize(torch.tensor(torch.load(feature_path)))

    coco_features = torch.load("pytorch_cache/features/coco_features_vitb32.pt")
    coco_mean = F.normalize(torch.tensor(coco_features["image_features"])).mean(0)
    if close_gap:
        features = features - coco_mean

    train_features = features[sampled_train_idxs]
    train_labels = labels[sampled_train_idxs]
    val_features = features[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Linear(512, 2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch_idx in range(25):
        train_one_epoch(train_dataloader, model, opt)
        metrics = evaluate(val_dataloader, model)
        print(epoch_idx, metrics)

    torch.save(model.state_dict(), "fairface_linear_model.pt")

    # ages = set([item["attributes"]["age"] for item in data])
    # races = set([item["attributes"]["race"] for item in data])
    # age_distribution = dict(Counter([item["attributes"]["age"] for item in data]).most_common())
    # age_percentage = {key: age_distribution[key] / sum(age_distribution.values()) for key in age_distribution}
    # race_distribution = Counter([item["attributes"]["race"] for item in data]).most_common()
    # print(age_distribution)
    # print(age_percentage)
    # print(race_distribution)

    # montana_age = {
    #     "0-2": 3.9 / 100,
    #     "3-9": 9.1 / 100,
    #     "10-19": 15.6 / 100,
    #     "20-29": 12.2 / 100,
    #     "30-39": 13.6 / 100,
    #     "40-49": 15.4 / 100,
    #     "50-59": 12.7 / 100,
    #     "60-69": 7.6 / 100,
    #     "more than 70": 9.9 / 100
    # }


def train_image_model_dspites(data_path: str, feature_path: str):
    data = [json.loads(line) for line in open(data_path)]
    labels = torch.tensor([item["label"] for item in data])

    all_train_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["color"] == "red"
    ]

    random.seed(1234)
    train_idxs = random.sample(all_train_idxs, int(len(all_train_idxs) * 0.9))
    val_idxs = [idx for idx in range(len(data)) if idx not in train_idxs]
    json.dump([train_idxs, val_idxs], open("train_val_idxs.json", "w"))
    print(len(train_idxs), len(val_idxs))

    features = F.normalize(torch.tensor(torch.load(feature_path)))

    train_features = features[train_idxs]
    train_labels = labels[train_idxs]
    val_features = features[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Linear(512, 3).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch_idx in range(25):
        train_one_epoch(train_dataloader, model, opt)
        metrics = evaluate(val_dataloader, model)
        print(epoch_idx, metrics)

    torch.save(model.state_dict(), "dspites_linear_model.pt")


def train_image_model_waterbird(
    data_path: str, feature_path: str, close_gap: bool, coco_norm: bool = True
):
    data = [json.loads(line) for line in open(data_path)]
    labels = torch.tensor([item["attributes"]["waterbird"] for item in data])

    train_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "train"
    ]
    val_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "val"
    ]

    features = F.normalize(torch.tensor(torch.load(feature_path)))

    coco_features = torch.load("pytorch_cache/features/coco_features_vitb32.pt")
    coco_mean = F.normalize(torch.tensor(coco_features["image_features"])).mean(0)
    if close_gap:
        if coco_norm:
            features = features - coco_mean
        else:
            features = features - features.mean(0)

    train_features = features[train_idxs]
    train_labels = labels[train_idxs]
    val_features = features[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Linear(512, 2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch_idx in range(25):
        train_one_epoch(train_dataloader, model, opt)
        metrics = evaluate(val_dataloader, model)
        print(epoch_idx, metrics)

    torch.save(model.state_dict(), f"waterbird_linear_model_gap={not close_gap}.pt")


def train_image_model_waterbird_dro(data_path: str, feature_path: str):
    data = [json.loads(line) for line in open(data_path)]
    for item in data:
        if (
            item["attributes"]["waterbird"] == 1
            and item["attributes"]["waterplace"] == 1
        ):
            item["attributes"]["group_idx"] = 0
        elif (
            item["attributes"]["waterbird"] == 1
            and item["attributes"]["waterplace"] == 0
        ):
            item["attributes"]["group_idx"] = 1
        elif (
            item["attributes"]["waterbird"] == 0
            and item["attributes"]["waterplace"] == 1
        ):
            item["attributes"]["group_idx"] = 2
        else:
            item["attributes"]["group_idx"] = 3

    labels = torch.tensor([item["attributes"]["waterbird"] for item in data])

    train_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "train"
    ]
    val_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "val"
    ]

    features = F.normalize(torch.tensor(torch.load(feature_path)))

    train_features = features[train_idxs]
    train_labels = labels[train_idxs]
    train_group_idxs = torch.tensor([item["attributes"]["group_idx"] for item in data])[
        train_idxs
    ]
    val_features = features[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset = TensorDataset(train_features, train_labels, train_group_idxs)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Linear(512, 2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    group_counts = torch.tensor(
        [
            np.sum(
                [
                    item["attributes"]["group_idx"] == i
                    for item in np.array(data)[train_idxs]
                ]
            )
            for i in range(4)
        ]
    ).cuda()
    print(group_counts)
    loss_dro = LossComputer(
        torch.nn.CrossEntropyLoss(reduce=False),
        True,
        0.2,
        0.1,
        np.array([0.0, 0.0, 0.0, 0.0]),
        0,
        0.01,
        False,
        False,
        4,
        group_counts,
        ["ww", "wl", "lw", "ll"],
    )
    for epoch_idx in range(25):
        train_one_epoch_dro(train_dataloader, model, opt, loss_dro)
        metrics = evaluate(val_dataloader, model)
        print(epoch_idx, metrics)

        preds, labels = metrics["preds"], metrics["labels"]
        subgroups = subgrouping(np.array(data)[val_idxs], ["waterbird", "waterplace"])
        subgroup_metrics = computing_subgroup_metrics(preds, labels, subgroups)  # type: ignore
        print(subgroup_metrics)

    torch.save(model.state_dict(), "waterbird_linear_model_dro.pt")


def train_image_model_fairface_dro(data_path: str, feature_path: str):
    data = [json.loads(line) for line in open(data_path)]
    labels = torch.tensor(
        [1 if item["attributes"]["gender"] == "Female" else 0 for item in data]
    )

    train_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "train"
    ]
    val_idxs = [
        i for i, item in enumerate(data) if item["attributes"]["split"] == "val"
    ]

    montana_race = {
        "White": 90.6 / 100,
        "Indian": 6.2 / 100,
        "East Asian": 0.5 / 100,
        "Black": 0.3 / 100,
    }
    race2idx = {
        "White": 0,
        "Indian": 1,
        "East Asian": 2,
        "Black": 3,
    }
    for item in data:
        race = item["attributes"]["race"]
        item["attributes"]["group_idx"] = race2idx[race] if race in race2idx else -1

    montana_race_norm = {
        key: montana_race[key] / montana_race["White"] for key in montana_race
    }
    print(montana_race_norm)

    sampled_train_idxs = []
    random.seed(1234)
    for idx in train_idxs:
        race = data[idx]["attributes"]["race"]
        if race in montana_race_norm:
            if random.random() <= montana_race_norm[race]:
                sampled_train_idxs.append(idx)

    # race_distribution = Counter([item["attributes"]["race"] for item in sampled_train_data]).most_common()

    features = F.normalize(torch.tensor(torch.load(feature_path)))

    train_features = features[sampled_train_idxs]
    train_labels = labels[sampled_train_idxs]
    train_group_idxs = torch.tensor([item["attributes"]["group_idx"] for item in data])[
        sampled_train_idxs
    ]
    val_features = features[val_idxs]
    val_labels = labels[val_idxs]

    train_dataset = TensorDataset(train_features, train_labels, train_group_idxs)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Linear(512, 2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    group_counts = torch.tensor(
        [
            np.sum(
                [
                    item["attributes"]["group_idx"] == i
                    for item in np.array(data)[train_idxs]
                ]
            )
            for i in range(4)
        ]
    ).cuda()
    print(group_counts)
    loss_dro = LossComputer(
        torch.nn.CrossEntropyLoss(reduce=False),
        True,
        0.2,
        0.1,
        np.array([0.0, 0.0, 0.0, 0.0]),
        0,
        0.01,
        False,
        False,
        4,
        group_counts,
        ["ww", "wl", "lw", "ll"],
    )
    for epoch_idx in range(25):
        train_one_epoch_dro(train_dataloader, model, opt, loss_dro)
        metrics = evaluate(val_dataloader, model)
        print(epoch_idx, metrics)

        preds, labels = metrics["preds"], metrics["labels"]
        subgroups = subgrouping(np.array(data)[val_idxs], ["race"])
        subgroup_metrics = computing_subgroup_metrics(preds, labels, subgroups)  # type: ignore
        print(subgroup_metrics)

    torch.save(model.state_dict(), "fairface_linear_model_dro.pt")


def train_waterbird():
    clip_model, transform = clip.load(name=CLIP_MODEL, device="cuda")
    clip_model = clip_model.float()
    model = Linear(clip_model.visual.output_dim, 2).cuda()

    image_dataset_train = AttributeDataset(
        path="data/Waterbird/processed_attribute_dataset/",
        filter_func=lambda x: x["attributes"]["split"] == "train",
        label_func=lambda x: x["attributes"]["waterbird"],
    )
    image_dataloader_train = create_dataloader(
        dataset=image_dataset_train, modality="image", transform=transform, shuffle=True
    )

    image_dataset_val = AttributeDataset(
        path="data/Waterbird/processed_attribute_dataset/",
        filter_func=lambda x: x["attributes"]["split"] == "val",
        label_func=lambda x: x["attributes"]["waterbird"],
    )
    image_dataloader_val = create_dataloader(
        dataset=image_dataset_val, modality="image", transform=transform, shuffle=False
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch_idx in range(25):
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
        for line in open("data/Waterbird/processed_attribute_dataset/attributes.jsonl")
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
    # train_waterbird()
    # eval_waterbird()
    # extract_image_features(sys.argv[1])
    # train_image_model_dspites(sys.argv[1], sys.argv[2])
    # train_image_model_waterbird(
    #     "data/Waterbird/processed_attribute_dataset/attributes.jsonl",
    #     "mmdebug/src/pytorch_cache/features/Waterbird_features_vitb32.pt",
    #     False,
    # )
    # train_image_model_waterbird(
    #     "data/Waterbird/processed_attribute_dataset/attributes.jsonl",
    #     "mmdebug/src/pytorch_cache/features/Waterbird_features_vitb32.pt",
    #     True,
    # )
    # train_image_model_fairface(
    #     "data/FairFace/processed_attribute_dataset/attributes.jsonl",
    #     "mmdebug/src/pytorch_cache/features/FairFace_features_vitb32.pt",
    #     False,
    # )
    # train_image_model_fairface(
    #     "data/FairFace/processed_attribute_dataset/attributes.jsonl",
    #     "mmdebug/src/pytorch_cache/features/FairFace_features_vitb32.pt",
    #     True,
    # )
    # train_image_model_waterbird_dro(
    #     "data/Waterbird/processed_attribute_dataset/attributes.jsonl",
    #     "mmdebug/src/pytorch_cache/features/Waterbird_features_vitb32.pt"
    # )
    train_image_model_fairface_dro(
        "data/FairFace/processed_attribute_dataset/attributes.jsonl",
        "mmdebug/src/pytorch_cache/features/FairFace_features_vitb32.pt",
    )
