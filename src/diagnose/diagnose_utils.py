import json
from typing import Callable, Dict, List, Optional, Tuple, Union

import clip  # type: ignore
import numpy as np
import torch
import torchvision  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from scipy.stats import pearsonr, spearmanr  # type: ignore

from datasets import ImageDataset, TextDataset, create_dataloader
from models import Linear
from trainer import run_one_epoch
from utils import computing_subgroup_metrics, subgrouping


def get_model_output(
    model: torch.nn.Module,
    clip_model: torch.nn.Module,
    transform: torchvision.transforms,
    image_data: List[Dict],
    generated_data: List[Dict],
    generated_data_modality: Optional[str] = "text",
) -> Tuple[Dict, Dict]:
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
        verbose=False,
    )

    generated_dataset: Union[TextDataset, ImageDataset]
    if generated_data_modality == "image":
        generated_dataset = ImageDataset(data=generated_data)
    elif generated_data_modality == "text":
        generated_dataset = TextDataset(data=generated_data)
    else:
        raise ValueError("Unknown modality")
    generated_dataloader = create_dataloader(
        dataset=generated_dataset, modality=generated_data_modality, transform=transform
    )
    generated_metrics = run_one_epoch(
        dataloader=generated_dataloader,
        model=model,
        clip_model=clip_model,
        modality=generated_data_modality,
        opt=None,
        epoch_idx=-1,
        eval=True,
        verbose=False,
    )
    return image_metrics, generated_metrics


def compute_correlation(
    data1_list: List, data2_list: List, visualization: bool = False
) -> None:
    assert len(data1_list) == len(data2_list)
    data1 = np.array(data1_list)
    data2 = np.array(data2_list)
    spearmanr_corr, spearmanr_pval = spearmanr(data1, data2)
    pearsonr_corr, pearsonr_pval = pearsonr(data1, data2)
    print(f"Spearman correlation: {spearmanr_corr:.4f} (p-value: {spearmanr_pval:.4f})")
    print(f"Pearson correlation: {pearsonr_corr:.4f} (p-value: {pearsonr_pval:.4f})")
    if visualization:
        plt.scatter(data1, data2)
        plt.xlabel("Image")
        plt.ylabel("Text")
        plt.show()


def compute_subgroup_correlation(
    image_data: List[Dict],
    image_metrics: Dict,
    text_data: List[Dict],
    text_metrics: Dict,
    fields: List[str],
    visualization: bool = False,
) -> Tuple[Dict, Dict, Dict]:
    image_subgroups = subgrouping(image_data, fields)
    image_instance_accs = np.array(image_metrics["preds"]) == np.array(
        image_metrics["labels"]
    )
    image_subgroup_accs = computing_subgroup_metrics(
        image_instance_accs, image_subgroups
    )

    text_subgroups = subgrouping(text_data, fields)
    text_instance_accs = np.array(text_metrics["preds"]) == np.array(
        text_metrics["labels"]
    )
    text_subgroup_accs = computing_subgroup_metrics(text_instance_accs, text_subgroups)

    text_instance_probs = torch.softmax(
        torch.tensor(text_metrics["logits"]), dim=1
    ).numpy()[np.arange(len(text_metrics["labels"])), text_metrics["labels"]]
    text_subgroup_probs = computing_subgroup_metrics(
        text_instance_probs, text_subgroups
    )

    print("Text Acc - Image Acc Correlation:")
    compute_correlation(
        [text_subgroup_accs[x] for x in image_subgroups],
        [image_subgroup_accs[x] for x in image_subgroups],
    )
    print("Text Prob - Image Acc Correlation:")
    compute_correlation(
        [text_subgroup_probs[x] for x in image_subgroups],
        [image_subgroup_accs[x] for x in image_subgroups],
    )
    return image_subgroup_accs, text_subgroup_accs, text_subgroup_probs


def compute_dataset_correlation(
    clip_model_name: str,
    linear_model_path: str,
    data_path: str,
    generated_data_path: str,
    filter_fn: Callable,
    label_fn: Callable,
    prepare_fn: Callable,
    fields: List[str],
) -> None:
    clip_model, transform = clip.load(name=clip_model_name, device="cuda")
    clip_model = clip_model.float()
    state_dict = torch.load(linear_model_path)
    n_class = state_dict["fc.weight"].shape[0]
    model = Linear(clip_model.visual.output_dim, n_class).cuda()
    model.load_state_dict(state_dict)

    image_data = [json.loads(line) for line in open(data_path)]
    image_data = [item for idx, item in enumerate(image_data) if filter_fn(idx, item)]
    for item in image_data:
        item["label"] = label_fn(item)

    text_data_concat = prepare_fn(data_path=data_path, input_type="concat")
    text_data_prompt = prepare_fn(data_path=data_path, input_type="prompt")
    text_data_ensemble = prepare_fn(data_path=data_path, input_type="ensemble")

    print("\nConcat:\n")
    image_metrics, text_metrics_concat = get_model_output(
        model, clip_model, transform, image_data, text_data_concat
    )
    compute_subgroup_correlation(
        image_data, image_metrics, text_data_concat, text_metrics_concat, fields=fields
    )

    print("\nPrompt:\n")
    image_metrics, text_metrics_prompt = get_model_output(
        model, clip_model, transform, image_data, text_data_prompt
    )
    compute_subgroup_correlation(
        image_data, image_metrics, text_data_prompt, text_metrics_prompt, fields=fields
    )

    print("\nEnsemble:\n")
    image_metrics, text_metrics_ensemble = get_model_output(
        model, clip_model, transform, image_data, text_data_ensemble
    )
    compute_subgroup_correlation(
        image_data,
        image_metrics,
        text_data_ensemble,
        text_metrics_ensemble,
        fields=fields,
    )

    print("\nImage Generation:\n")
    generated_image_data = [json.loads(line) for line in open(generated_data_path)]
    image_metrics, generated_image_metrics = get_model_output(
        model,
        clip_model,
        transform,
        image_data,
        generated_image_data,
        generated_data_modality="image",
    )
    compute_subgroup_correlation(
        image_data,
        image_metrics,
        generated_image_data,
        generated_image_metrics,
        fields=fields,
    )


def discover_slices(
    clip_model_name: str,
    linear_model_path: str,
    data_path: str,
    filter_fn: Callable,
    label_fn: Callable,
    prepare_fn: Callable,
    fields: List[str],
) -> List:
    clip_model, transform = clip.load(name=clip_model_name, device="cuda")
    clip_model = clip_model.float()
    state_dict = torch.load(linear_model_path)
    n_class = state_dict["fc.weight"].shape[0]
    model = Linear(clip_model.visual.output_dim, n_class).cuda()
    model.load_state_dict(state_dict)

    image_data = [json.loads(line) for line in open(data_path)]
    image_data = [item for idx, item in enumerate(image_data) if filter_fn(idx, item)]
    for item in image_data:
        item["label"] = label_fn(item)

    text_data = prepare_fn(data_path=data_path, input_type="ensemble")

    image_metrics, text_metrics = get_model_output(
        model, clip_model, transform, image_data, text_data
    )
    (
        image_subgroup_accs,
        text_subgroup_accs,
        text_subgroup_probs,
    ) = compute_subgroup_correlation(
        image_data,
        image_metrics,
        text_data,
        text_metrics,
        fields=fields,
    )
    subgroup_accs = {
        key: (
            image_subgroup_accs[key],
            text_subgroup_accs[key],
            text_subgroup_probs[key],
        )
        for key in image_subgroup_accs
    }
    return sorted(subgroup_accs.items(), key=lambda x: x[1][0])
