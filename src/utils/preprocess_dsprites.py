import json
import os
import random

from PIL import Image, ImageDraw  # type: ignore


def triangle_img(color="white", angle=0, scale=1.0):
    triangle_img = Image.new("RGB", (100, 100), "black")
    triangle_draw = ImageDraw.Draw(triangle_img)
    triangle_draw.polygon([(20, 24), (50, 76), (80, 24)], fill=color)
    return triangle_img.rotate(angle).resize((int(100 * scale), int(100 * scale)))


def square_img(color="white", angle=0, scale=1.0):
    square_img = Image.new("RGB", (100, 100), "black")
    square_draw = ImageDraw.Draw(square_img)
    square_draw.polygon([(20, 20), (80, 20), (80, 80), (20, 80)], fill=color)
    return square_img.rotate(angle).resize((int(100 * scale), int(100 * scale)))


def circle_img(color="white", angle=0, scale=1.0):
    circle_img = Image.new("RGB", (100, 100), "black")
    circle_draw = ImageDraw.Draw(circle_img)
    circle_draw.ellipse([(20, 20), (80, 80)], fill=color)
    return circle_img.rotate(angle).resize((int(100 * scale), int(100 * scale)))


def create_dsprites_dataset():
    random.seed(1234)
    path = "data/dSprites"

    os.mkdir(f"{path}/processed_attribute_dataset")
    os.mkdir(f"{path}/processed_attribute_dataset/images")

    data = []

    for i in range(10000):
        img = Image.new("RGB", (224, 224), "black")
        angle = random.randint(0, 360)

        poses = [
            [0, 28, 0, 28],
            [112, 140, 112, 140],
            [112, 140, 0, 28],
            [0, 28, 112, 140],
        ]
        pos_idx = random.randint(0, 3)
        x = random.randint(poses[pos_idx][0], poses[pos_idx][1])
        y = random.randint(poses[pos_idx][2], poses[pos_idx][3])

        scale = random.choice([0.5, 1.0, 1.5])
        color = random.choice(
            [
                "red",
                "pink",
                "orange",
                "green",
                "cyan",
                "blue",
                "purple",
                "yellow",
                "violet",
                "brown",
            ]
        )
        label = random.choice(["triangle", "square", "circle"])

        if label == "triangle":
            img.paste(triangle_img(color, angle, scale), (x, y))
        elif label == "square":
            img.paste(square_img(color, angle, scale), (x, y))
        elif label == "circle":
            img.paste(circle_img(color, angle, scale), (x, y))

        data.append(
            {
                "image": f"{path}/processed_attribute_dataset/images/{i}.png",
                "label": ["triangle", "square", "circle"].index(label),
                "attributes": {
                    "color": color,
                    "angle": angle,
                    "scale": scale,
                    "position": pos_idx,
                    "concrete_position": [x, y],
                    "label": label,
                },
            }
        )
        img.save(f"{path}/processed_attribute_dataset/images/{i}.png")

    with open(f"{path}/processed_attribute_dataset/attributes.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


create_dsprites_dataset()
