import clip  # type: ignore
import torch

from datasets import AttributeDataset, TextDataset, create_dataloader
from models import Linear
from trainer import run_one_epoch


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
    main()
