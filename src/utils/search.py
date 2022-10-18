from collections import defaultdict
from typing import Dict, List

import numpy as np


def subgrouping(data: List[Dict], fields: List[str]) -> Dict:
    assert all(field in data[0]["attributes"] for field in fields), "Invalid fields"
    subgroups = defaultdict(list)
    for i, x in enumerate(data):
        subgroups[tuple([(field, x["attributes"][field]) for field in fields])].append(
            i
        )
    return dict(sorted(subgroups.items()))


def computing_subgroup_metrics(instance_metrics: np.ndarray, subgroups: Dict) -> Dict:
    subgroup_metrics = {x: instance_metrics[subgroups[x]].mean() for x in subgroups}
    return subgroup_metrics


if __name__ == "__main__":
    data = [
        {
            "attributes": {
                "place": "place1",
                "species": "species1",
            }
        },
        {
            "attributes": {
                "place": "place2",
                "species": "species2",
            }
        },
        {
            "attributes": {
                "place": "place3",
                "species": "species3",
            }
        },
    ]
    fields = ["place", "species"]
    print(subgrouping(data, fields))
