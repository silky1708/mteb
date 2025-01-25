from __future__ import annotations

import json

from tqdm import tqdm

import mteb

tasks = mteb.get_tasks(
    task_types=["ImageClassification"],
)

for task in tqdm(tasks, total=len(tasks)):
    task.calculate_metadata_metrics()
    break
exit()

for task in tqdm(tasks, total=len(tasks)):
    stats_dict = task.metadata.descriptive_stats
    import pdb

    pdb.set_trace()
    descriptive_stat_path = task.metadata.descriptive_stat_path
    # if descriptive_stat_path.exists():
    with descriptive_stat_path.open("w") as f:
        json.dump(stats_dict, f)
    break
