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


def computing_subgroup_metrics(
    preds: List[int], labels: List[int], subgroups: Dict
) -> Dict:
    assert len(preds) == len(labels)
    subgroup_metrics = {}
    for x in subgroups:
        subgroup_preds = np.array(preds)[subgroups[x]]
        subgroup_labels = np.array(labels)[subgroups[x]]
        subgroup_metrics[x] = (subgroup_preds == subgroup_labels).mean()
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
