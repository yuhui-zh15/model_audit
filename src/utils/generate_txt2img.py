import argparse
import json
import os

from diffusers import StableDiffusionPipeline  # type: ignore

# make sure you're logged in with `huggingface-cli login`
from torch import autocast


def main(args):

    # check if output dir exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # create diffusion model
    model = StableDiffusionPipeline.from_pretrained(
        args.diffusion_model, use_auth_token=True
    ).to(args.device)

    # read in prompts
    prompts = [json.loads(line) for line in open(args.prompt_path)]
    name = args.prompt_path.split("/")[-1].split(".")[0]

    image_data = []
    for i, prompt in enumerate(prompts):

        new_data = prompt.copy()
        for j in range(args.n_img):

            # generate and save image
            with autocast(args.device):
                image = model(new_data["text"])["sample"][0]
            img_path = os.path.join(args.output_dir, f"{name}_{i}_{j}.png")
            image.save(img_path)

            # save image path to output
            new_data["image"] = img_path
            image_data.append(prompt)

    # save new dataset
    dataset_path = os.path.join(args.output_dir, f"{name}_generated_images.jsonl")
    with open(dataset_path, "w") as f:
        for item in image_data:
            f.write(json.dumps(item) + "\n")
    print(f"Generated dataset saved at {dataset_path}")


if __name__ == "__main__":

    # parse args and configs
    parser = argparse.ArgumentParser("Genearte image from text")
    parser.add_argument(
        "--prompt_path",
        default="waterbird_text_data.jsonl",
        type=str,
        help="Confounder to examine",
    )
    parser.add_argument(
        "--n_img", default=50, type=int, help="Number of images per prompt"
    )
    parser.add_argument(
        "--output_dir",
        default="data/GeneratedWaterBird",
        type=str,
        help="Confounder to examine",
    )
    parser.add_argument(
        "--diffusion_model",
        default="CompVis/stable-diffusion-v1-4",
        type=str,
        help="Confounder to examine",
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Confounder to examine"
    )
    args = parser.parse_args()

    main(args)
