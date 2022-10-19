import itertools
import json

from utils import openai_imagenet_template


def prepare_waterbird(data_path: str, input_type: str) -> list:
    def concat(x: dict) -> list:
        return [f"{x['species']}, {x['place']}.".lower()]

    def prompt(x: dict) -> list:
        return [f"a photo of a {x['species']} in the {x['place']}.".lower()]

    def ensemble(x: dict) -> list:
        return [
            pg(f"{x['species']} in the {x['place']}").lower()
            for pg in openai_imagenet_template
        ]

    data = [json.loads(line) for line in open(data_path)]

    attributes = {
        "place": set([x["attributes"]["place"] for x in data]),
        "species": set([x["attributes"]["species"] for x in data]),
    }
    attributes_combinations = [
        dict(zip(attributes, x))
        for x in sorted(itertools.product(*attributes.values()))
    ]
    species_to_label = {
        x["attributes"]["species"]: x["attributes"]["waterbird"] for x in data
    }
    places_to_label = {
        x["attributes"]["place"]: x["attributes"]["waterplace"] for x in data
    }

    generator = {
        "concat": concat,
        "prompt": prompt,
        "ensemble": ensemble,
    }[input_type]
    text_data = [
        {
            "text": text,
            "label": species_to_label[x["species"]],
            "attributes": {
                "waterbird": species_to_label[x["species"]],
                "waterplace": places_to_label[x["place"]],
                "species": x["species"],
                "place": x["place"],
            },
        }
        for x in attributes_combinations
        for text in generator(x)
    ]
    return text_data


def prepare_fairface(data_path: str, input_type: str) -> list:
    def concat(x: dict) -> list:
        return [f"{x['age']}, {x['race']}, {x['gender']}.".replace("_", " ").lower()]

    def prompt(x: dict) -> list:
        return [
            f"a face of a {x['race']} {age_description[x['age']][gender_to_idx[x['gender']]]}.".replace(
                "_", " "
            ).lower()
        ]

    def ensemble(x: dict) -> list:
        return [
            pg(
                f"face of a {x['race']} {age_description[x['age']][gender_to_idx[x['gender']]]}".replace(
                    "_", " "
                ).lower()
            )
            for pg in openai_imagenet_template
        ]

    data = [json.loads(line) for line in open(data_path)]

    attributes = {
        "age": set([x["attributes"]["age"] for x in data]),
        "race": set([x["attributes"]["race"] for x in data]),
        "gender": set([x["attributes"]["gender"] for x in data]),
    }
    attributes_combinations = [
        dict(zip(attributes, x))
        for x in sorted(itertools.product(*attributes.values()))
    ]
    age_description = {
        "0-2": ["infant boy", "infant girl"],
        "3-9": ["little boy", "little girl"],
        "10-19": ["teenage boy", "teenage girl"],
        "20-29": ["young man", "young woman"],
        "30-39": ["adult man", "adult woman"],
        "40-49": ["middle-aged man", "middle-aged woman"],
        "50-59": ["senior man", "senior woman"],
        "60-69": ["elderly man", "elderly woman"],
        "more than 70": ["very old man", "very old woman"],
    }
    gender_to_idx = {
        "Male": 0,
        "Female": 1,
    }

    generator = {
        "concat": concat,
        "prompt": prompt,
        "ensemble": ensemble,
    }[input_type]
    text_data = [
        {
            "text": text,
            "label": 1 if x["gender"] == "Female" else 0,
            "attributes": {"age": x["age"], "race": x["race"], "gender": x["gender"]},
        }
        for x in attributes_combinations
        for text in generator(x)
    ]
    return text_data


def prepare_dsprites(data_path: str, input_type: str) -> list:
    def concat(x: dict) -> list:
        return [
            f"{['small', 'medium', 'large'][x['concrete_scale']]}, {x['color']}, {['square', 'triangle'][x['label']]}.".lower()  # noqa: E501
        ]

    def prompt(x: dict) -> list:
        return [
            f"{['small', 'medium', 'large'][x['concrete_scale']]} {x['color']} {['square', 'triangle'][x['label']]}.".lower()  # noqa: E501
        ]

    def ensemble(x: dict) -> list:
        return [
            pg(
                f"{['small', 'medium', 'large'][x['concrete_scale']]} {x['color']} {['square', 'triangle'][x['label']]}"
            ).lower()
            for pg in openai_imagenet_template
        ]

    data = [json.loads(line) for line in open(data_path)]

    for item in data:
        if item["attributes"]["scale"] < 0.9:
            item["attributes"]["concrete_scale"] = 0
        elif item["attributes"]["scale"] > 1.1:
            item["attributes"]["concrete_scale"] = 2
        else:
            item["attributes"]["concrete_scale"] = 1

    attributes = {
        "color": set([x["attributes"]["color"] for x in data]),
        "label": set([x["attributes"]["label"] for x in data]),
        "concrete_scale": set([x["attributes"]["concrete_scale"] for x in data]),
    }
    attributes_combinations = [
        dict(zip(attributes, x))
        for x in sorted(itertools.product(*attributes.values()))
    ]

    generator = {
        "concat": concat,
        "prompt": prompt,
        "ensemble": ensemble,
    }[input_type]
    text_data = [
        {
            "text": text,
            "label": x["label"],
            "attributes": {
                "color": x["color"],
                "label": x["label"],
                "concrete_scale": x["concrete_scale"],
            },
        }
        for x in attributes_combinations
        for text in generator(x)
    ]
    return text_data


if __name__ == "__main__":
    text_data = prepare_waterbird(
        data_path="../data/Waterbird/processed_attribute_dataset/attributes.jsonl",
        input_type="concat",
    )
    print(text_data[0], len(text_data))
    text_data = prepare_waterbird(
        data_path="../data/Waterbird/processed_attribute_dataset/attributes.jsonl",
        input_type="prompt",
    )
    print(text_data[0], len(text_data))
    text_data = prepare_waterbird(
        data_path="../data/Waterbird/processed_attribute_dataset/attributes.jsonl",
        input_type="ensemble",
    )
    print(text_data[0], len(text_data))

    text_data = prepare_fairface(
        data_path="../data/FairFace/processed_attribute_dataset/attributes.jsonl",
        input_type="concat",
    )
    print(text_data[0], len(text_data))
    text_data = prepare_fairface(
        data_path="../data/FairFace/processed_attribute_dataset/attributes.jsonl",
        input_type="prompt",
    )
    print(text_data[0], len(text_data))
    text_data = prepare_fairface(
        data_path="../data/FairFace/processed_attribute_dataset/attributes.jsonl",
        input_type="ensemble",
    )
    print(text_data[0], len(text_data))

    text_data = prepare_dsprites(
        data_path="../data/TriangleSquare/processed_attribute_dataset/attributes.jsonl",
        input_type="concat",
    )
    print(text_data[0], len(text_data))
    text_data = prepare_dsprites(
        data_path="../data/TriangleSquare/processed_attribute_dataset/attributes.jsonl",
        input_type="prompt",
    )
    print(text_data[0], len(text_data))
    text_data = prepare_dsprites(
        data_path="../data/TriangleSquare/processed_attribute_dataset/attributes.jsonl",
        input_type="ensemble",
    )
    print(text_data[0], len(text_data))
