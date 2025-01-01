# based on: create_slurm_jobs.py
"""Creates slurm jobs for running models on all tasks"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import mteb
from mteb.benchmarks import MTEB_MAIN_EN


def create_slurm_job_file(
    model_name: str,
    task_name: str,
    results_folder: Path,
    slurm_prefix: str,
    slurm_jobs_folder: Path,
) -> Path:
    """Create slurm job file for running a model on a task"""
    slurm_job = f"{slurm_prefix}\n"
    slurm_job += f"mteb run -m {model_name} -t {task_name} --output_folder {results_folder.resolve()} --co2_tracker true --batch_size 16"

    model_path_name = model_name.replace("/", "__")

    slurm_job_file = slurm_jobs_folder / f"{model_path_name}_{task_name}.sh"
    with open(slurm_job_file, "w") as f:
        f.write(slurm_job)
    return slurm_job_file


def create_slurm_job_files(
    model_names: list[str],
    tasks: Iterable[mteb.AbsTask],
    results_folder: Path,
    slurm_prefix: str,
    slurm_jobs_folder: Path,
) -> list[Path]:
    """Create slurm job files for running models on all tasks"""
    slurm_job_files = []
    for model_name in model_names:
        for task in tasks:
            slurm_job_file = create_slurm_job_file(
                model_name,
                task.metadata.name,
                results_folder,
                slurm_prefix,
                slurm_jobs_folder,
            )
            slurm_job_files.append(slurm_job_file)
    return slurm_job_files


def run_slurm_jobs(files: list[Path]) -> None:
    """Run slurm jobs based on the files provided"""
    for file in files:
        subprocess.run(["sbatch", file])


if __name__ == "__main__":
    # SHOULD BE UPDATED
    slurm_prefix = """#!/bin/bash
#SBATCH --job-name=mteb
#SBATCH --nodes=1
#SBATCH --partition=a3low
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 72:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name
#SBATCH --exclusive
"""

    project_root = Path(__file__).parent / ".." / ".." / ".."
    results_folder = project_root / "results"
    results_folder = Path("/data/niklas/results_speed")
    slurm_jobs_folder = Path(__file__).parent / "slurm_jobs"

    model_names = [
        "GritLM/GritLM-7B",
        "intfloat/multilingual-e5-large",
    ]

    # expanding to a full list of tasks
    task_names = [
        "ArXivHierarchicalClusteringP2P",
        "ArXivHierarchicalClusteringS2S",
        "BiorxivClusteringP2P.v2",
        "BiorxivClusteringS2S.v2",
        "MedrxivClusteringP2P.v2",
        "MedrxivClusteringS2S.v2",
        "RedditClustering.v2",
        "RedditClusteringP2P.v2",
        "STS22.v2",
        "StackExchangeClustering.v2",
        "StackExchangeClusteringP2P.v2",
        "SummEvalSummarization.v2",
        "TwentyNewsgroupsClustering.v2"
    ]

    tasks = mteb.get_tasks(tasks=task_names) # MTEB_MAIN_EN.tasks

    slurm_jobs_folder.mkdir(exist_ok=True)
    files = create_slurm_job_files(
        model_names, tasks, results_folder, slurm_prefix, slurm_jobs_folder
    )
    run_slurm_jobs(files)