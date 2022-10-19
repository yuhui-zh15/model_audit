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


def get_concrete_scale(scale):
    if scale < 0.9:
        return 0
    elif scale > 1.1:
        return 2
    else:
        return 1


def create_triangle_square_classification_dataset():
    random.seed(1234)
    path = "data/TriangleSquare"

    os.mkdir(f"{path}/processed_attribute_dataset")
    os.mkdir(f"{path}/processed_attribute_dataset/images")

    data = []

    for i in range(10000):
        img = Image.new("RGB", (224, 224), "black")
        angle = random.randint(0, 360)
        scale = random.uniform(0.8, 1.2)

        poses = [
            [0, 28, 0, 28],
            [112, 140, 112, 140],
            [112, 140, 0, 28],
            [0, 28, 112, 140],
        ]
        pos_idx = random.randint(0, 3)
        x = random.randint(poses[pos_idx][0], poses[pos_idx][1])
        y = random.randint(poses[pos_idx][2], poses[pos_idx][3])

        color = random.choice(["red", "pink", "orange", "green", "cyan", "blue"])

        if random.random() < 0.5:
            img.paste(triangle_img(color, angle, scale), (x, y))
            label = 1
        else:
            img.paste(square_img(color, angle, scale), (x, y))
            label = 0
        data.append(
            {
                "image": f"{path}/processed_attribute_dataset/images/{i}.png",
                "attributes": {
                    "color": color,
                    "angle": angle,
                    "scale": scale,
                    "position": pos_idx,
                    "concrete_position": [x, y],
                    "concrete_scale": get_concrete_scale(scale),
                    "label": label,
                },
            }
        )
        img.save(f"{path}/processed_attribute_dataset/images/{i}.png")

    with open(f"{path}/processed_attribute_dataset/attributes.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


create_triangle_square_classification_dataset()
