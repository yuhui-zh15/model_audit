import argparse
import itertools
import json
from pprint import pprint

import clip  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt  # type: ignore
from scipy.stats import pearsonr, spearmanr  # type: ignore

from datasets import ImageDataset, TextDataset, create_dataloader
from models import Linear
from trainer import run_one_epoch
from utils import computing_subgroup_metrics, subgrouping

DATASET_PATHS = {
    "waterbird": "/pasteur/u/yuhuiz/data/Waterbird/processed_attribute_dataset/attributes.jsonl",
    "waterbird_generated": "/pasteur/u/yuhuiz/data/GeneratedWaterBird/waterbird_text_data_generated_images_n=20.jsonl",
    "triangelsquare": "/pasteur/u/yuhuiz/data/TriangleSquare/processed_attribute_dataset/attributes.jsonl",
    "fairface": "/pasteur/u/yuhuiz/data/FairFace/processed_attribute_dataset/attributes.jsonl",
    "celeba": "/pasteur/u/yuhuiz/data/CelebA/processed_attribute_dataset/attributes.jsonl",
}

# TODO: will remove
DATASET_ATTRIBUTES = {
    "waterbird": ["species", "place_raw", "place", "waterbird", "waterplace"],
    "triangelsquare": [
        "color",
        "angle",
        "scale",
        "position",
        "concrete_position",
        "label",
    ],
    "fairface": ["age", "gender", "race", "service_test"],
    "celeba": [
        "Attractive",
        "Male",
        "5_o_Clock_Shadow",
        "Arched_Eyebrows",
        "Bags_Under_Eyes",
        "Bald",
        "Bangs",
        "Big_Lips",
        "Big_Nose",
        "Black_Hair",
        "Blond_Hair",
        "Blurry",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Chubby",
        "Double_Chin",
        "Eyeglasses",
        "Goatee",
        "Gray_Hair",
        "Heavy_Makeup",
        "High_Cheekbones",
        "Mouth_Slightly_Open",
        "Mustache",
        "Narrow_Eyes",
        "No_Beard",
        "Oval_Face",
        "Pale_Skin",
        "Pointy_Nose",
        "Receding_Hairline",
        "Rosy_Cheeks",
        "Sideburns",
        "Smiling",
        "Straight_Hair",
        "Wavy_Hair",
        "Wearing_Earrings",
        "Wearing_Hat",
        "Wearing_Lipstick",
        "Wearing_Necklace",
        "Wearing_Necktie",
        "Young",
    ],
}


def load_clip_model(
    model_name: str, device: str = "cuda", n_cls: int = 2, ckpt_path: str = None
):

    clip_model, transform = clip.load(name=model_name, device=device)
    clip_model = clip_model.float()
    model = Linear(clip_model.visual.output_dim, n_cls).cuda()

    if ckpt_path is not None:
        state_dict = torch.load("waterbird_linear_model.pt")
        model.load_state_dict(state_dict)

    return clip_model, model, transform


def filter_fn(x, cat, val):
    return x["attributes"][cat] == val


def label_fn(x, label):
    return x["attributes"][label]


def get_img_dataloader(args, transform):

    # get dataset
    data_path = DATASET_PATHS[args.dataset]
    image_data = [json.loads(line) for line in open(data_path)]

    if args.filter_category is not None:
        image_data = [
            x
            for x in image_data
            if filter_fn(x, args.filter_category, args.filter_value)
        ]

    # set label
    for item in image_data:
        item["label"] = label_fn(item, args.label)

    # create dataloader
    image_dataset = ImageDataset(data=image_data)
    image_dataloader = create_dataloader(
        dataset=image_dataset, modality="image", transform=transform
    )
    return image_data, image_dataloader


def get_text_dataloader(args, image_data):
    fg_conf = args.finegrain_confounder
    fg_lab = args.finegrain_label

    # create attribute combinations based on fine-grained categories
    attributes = {
        args.finegrain_confounder: set([x["attributes"][fg_conf] for x in image_data]),
        args.finegrain_label: set([x["attributes"][fg_lab] for x in image_data]),
    }
    attributes_combinations = [
        dict(zip(attributes, x)) for x in itertools.product(*attributes.values())
    ]

    # mappings from finegrain to corsegrain
    target_to_label = {
        x["attributes"][fg_lab]: x["attributes"][args.label] for x in image_data
    }
    confounder_to_label = {
        x["attributes"][fg_conf]: x["attributes"][args.confounder] for x in image_data
    }

    # create combinations of text prompts
    text_data = [
        {
            "text": f"a photo of a {x[fg_lab]} {args.preposition} {x[fg_conf]}.",
            "label": target_to_label[x[fg_lab]],
            "attributes": {
                args.label: target_to_label[x[fg_lab]],
                args.confounder: confounder_to_label[x[fg_conf]],
                args.finegrain_confounder: x[fg_conf],
                args.finegrain_label: x[fg_lab],
            },
        }
        for x in attributes_combinations
    ]

    # create dataloader
    text_dataset = TextDataset(data=text_data)
    text_dataloader = create_dataloader(dataset=text_dataset, modality="text")

    return text_data, text_dataloader


def compute_group_analysis(
    dataloader, clip_model, model, data, label, confounder, modality="image"
):

    print(f"INPUT MODALITY: {modality}")

    metrics = run_one_epoch(
        dataloader=dataloader,
        model=model,
        clip_model=clip_model,
        modality=modality,
        opt=None,
        epoch_idx=-1,
        eval=True,
        verbose=True,
    )
    preds, labels = metrics["preds"], metrics["labels"]
    subgroups = subgrouping(data, (label, confounder))
    subgroup_metrics = computing_subgroup_metrics(preds, labels, subgroups)
    return subgroup_metrics, metrics


def compute_and_plot_correlation(values, plot_path):

    # compute coorelation
    values = np.array(values)
    pearson = pearsonr(values[:, 0], values[:, 1])
    spearman = spearmanr(values[:, 0], values[:, 1])
    print(f"Pearson correlation: {pearson}")
    print(f"Spearman correlation: {spearman}")

    # plot
    plt.scatter(values[:, 0], values[:, 1], alpha=0.3, s=10)
    plt.savefig(plot_path)

    return pearson, spearman


def main(args):

    # get model
    clip_model, model, transform = load_clip_model(
        model_name=args.clip_model, device=args.device, ckpt_path=args.ckpt_path
    )

    # create image dataset
    image_data, image_dataloader = get_img_dataloader(args, transform)

    # create image dataset
    text_data, text_dataloader = get_text_dataloader(args, image_data)

    # DEBUG #1: Group Analysis
    print("=" * 80 + "Group Analysis" + "=" * 80)
    img_subgroup_metrics, _ = compute_group_analysis(
        image_dataloader,
        clip_model,
        model,
        image_data,
        args.label,
        args.confounder,
        modality="image",
    )
    pprint(sorted(img_subgroup_metrics.items(), key=lambda x: x[1]))

    print("-" * 80)
    txt_subgroup_metrics, _ = compute_group_analysis(
        text_dataloader,
        clip_model,
        model,
        text_data,
        args.label,
        args.confounder,
        modality="text",
    )
    pprint(sorted(txt_subgroup_metrics.items(), key=lambda x: x[1]))
    print("\n")

    # DEBUG #2: Sub-group Analysis
    print("=" * 80 + "Sub-group Analysis" + "=" * 80)
    img_fg_subgroup_metrics, img_fg_metrics = compute_group_analysis(
        image_dataloader,
        clip_model,
        model,
        image_data,
        args.finegrain_label,
        args.finegrain_confounder,
        modality="image",
    )
    pprint(sorted(img_fg_subgroup_metrics.items(), key=lambda x: x[1])[:5])
    print("...")
    pprint(sorted(img_fg_subgroup_metrics.items(), key=lambda x: x[1])[-5:])

    print("-" * 80)
    txt_fg_subgroup_metrics, txt_fg_metrics = compute_group_analysis(
        text_dataloader,
        clip_model,
        model,
        text_data,
        args.finegrain_label,
        args.finegrain_confounder,
        modality="text",
    )
    pprint(sorted(txt_fg_subgroup_metrics.items(), key=lambda x: x[1])[:5])
    print("...")
    pprint(sorted(txt_fg_subgroup_metrics.items(), key=lambda x: x[1])[-5:])
    print("\n")

    # DEBUG #2b: correlation
    print("=" * 80 + "Correlation Analysis" + "=" * 80)
    print("Metric: Acc")
    accs = [
        (txt_fg_subgroup_metrics[x], img_fg_subgroup_metrics[x])
        for x in img_fg_subgroup_metrics
    ]
    acc_plot_path = f"{args.dataset}_acc.png"
    acc_pearson, acc_spearman = compute_and_plot_correlation(accs, acc_plot_path)

    print("-" * 80)
    print("Metric: Probs")
    text_logits = txt_fg_metrics["logits"]
    text_probs = torch.softmax(torch.tensor(text_logits), dim=1).numpy().tolist()
    text_subgroup_probs = {
        (
            (args.finegrain_label, x["attributes"][args.finegrain_label]),
            (args.finegrain_confounder, x["attributes"][args.finegrain_confounder]),
        ): text_probs[i][x["attributes"][args.label]]
        for i, x in enumerate(text_data)
    }
    probs = [
        (text_subgroup_probs[x], img_fg_subgroup_metrics[x])
        for x in img_fg_subgroup_metrics
    ]
    prob_plot_path = f"{args.dataset}_probs.png"
    pearson, spearman = compute_and_plot_correlation(probs, prob_plot_path)
    print("\n")

    # DEBUG 3: Direct influence
    pprint("=" * 80 + "Direct influence" + "=" * 80)
    attributes = {
        args.finegrain_confounder: set(
            [x["attributes"][args.finegrain_confounder] for x in image_data]
        ),
        args.finegrain_label: set(
            [x["attributes"][args.finegrain_label] for x in image_data]
        ),
    }
    attribute_list = list(attributes[args.finegrain_confounder])
    attribute_embeddings = F.normalize(
        clip_model.encode_text(clip.tokenize(attribute_list).cuda())
    )
    probs = torch.softmax(model(attribute_embeddings), dim=1)
    print(f"For {args.label} (label = 1), the most similar attributes are:")
    print(
        sorted(
            zip(attribute_list, probs[:, 1].tolist()), key=lambda x: x[1], reverse=True
        )
    )
    print("\n")

    # DEBUG 4: Direct influence
    pprint("=" * 80 + "Aggregated influence" + "=" * 80)
    fg_conf_list = list(attributes[args.finegrain_confounder])
    fg_label_list = list(attributes[args.finegrain_label])
    shapley_values = {}
    for conf in fg_conf_list:
        prompts = [f"a photo of a {label}." for label in fg_label_list] + [
            f"a photo of a {label} in the {conf}." for label in fg_label_list
        ]
        with torch.no_grad():
            inputs = clip.tokenize(prompts).cuda()
            embeddings = F.normalize(clip_model.encode_text(inputs))
            probs = torch.softmax(model(embeddings), dim=1).cpu()

        shapley = (
            probs[len(fg_label_list) :, 1]  # noqa: E203
            - probs[: len(fg_label_list), 1]  # noqa: E203
        ).mean()
        print(f"Shapley value of {conf} for {args.label} = {shapley}")
        shapley_values[conf] = shapley

    # TODO:
    # Store results in dataframe


if __name__ == "__main__":

    # parse args and configs
    parser = argparse.ArgumentParser("Debug Analysis")
    parser.add_argument(
        "--clip_model",
        default="ViT-B/32",
        type=str,
        help="Version of CLIP model to load",
    )
    parser.add_argument(
        "--ckpt_path",
        default="waterbird_linear_model.pt",
        type=str,
        help="Path to pretrained checkpoint",
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Device to place mode"
    )
    parser.add_argument(
        "--dataset", default="waterbird", type=str, help="Device to place mode"
    )
    parser.add_argument(
        "--filter_category",
        default=None,
        type=str,
        help="Category to filter dataset on",
    )
    parser.add_argument(
        "--filter_value", default=None, type=str, help="Value to filter dataset with"
    )
    parser.add_argument(
        "--label", default="waterbird", type=str, help="Attribute to use as label "
    )
    parser.add_argument(
        "--confounder", default="waterplace", type=str, help="Confounder to examine"
    )
    parser.add_argument(
        "--finegrain_label",
        default="species",
        type=str,
        help="Attribute to use as label ",
    )
    parser.add_argument(
        "--finegrain_confounder",
        default="place",
        type=str,
        help="Confounder to examine",
    )
    parser.add_argument(
        "--preposition", default="in the", type=str, help="Confounder to examine"
    )
    args = parser.parse_args()

    main(args)
