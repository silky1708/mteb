"""Creates slurm jobs for running models on all tasks"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

import mteb


def create_slurm_job_file(
    model_name: str,
    task_name: str,
    results_folder: Path,
    slurm_prefix: str,
    slurm_jobs_folder: Path,
) -> Path:
    """Create slurm job file for running a model on a task"""
    slurm_job = f"{slurm_prefix}\n"
    # slurm_job += f"mteb run -m {model_name} -t {task_name} --output_folder {results_folder.resolve()} --co2_tracker true --batch_size 64"
    model_name_without_slash = model_name.replace("/", "__")
    slurm_job += (
        f"mteb run -m {model_name} -t {task_name} --output_folder {results_folder.resolve()} "
        f"--co2_tracker true --batch_size 4 || (mkdir -p /data/niklas/mteb/failures && "
        f"echo '{model_name}_{task_name}' >> /data/niklas/mteb/failures/{model_name_without_slash}_{task_name}.txt)"
    )


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
        print(f"Preparing to run {file}")
        # Only run if squeue --me | wc -l has less than 250 jobs
        # Technically scheduler allows more than 250 concurrent jobs but usually at 250 all GPUs are busy anyways so it keeps the queue cleaner
        while int(
            subprocess.run(["squeue", "--me"], capture_output=True, text=True).stdout.count("\n")
        ) > 165:
            print("Waiting for jobs to finish...")
            subprocess.run(["sleep", "10"])
        subprocess.run(["sbatch", file])


if __name__ == "__main__":
    # Update prefixes to match your cluster configuration
    slurm_prefix_8gpus = """#!/bin/bash
#SBATCH --job-name=gritkto
#SBATCH --nodes=1
#SBATCH --partition=a3low
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 30-00:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name
"""
    slurm_prefix = """#!/bin/bash
#SBATCH --job-name=gritkto
#SBATCH --nodes=1
#SBATCH --partition=a3mixedlow
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 30-00:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name
"""
##SBATCH --exclusive
    project_root = Path(__file__).parent / ".." / ".." / ".."
    # results_folder = project_root / "results"
    results_folder = Path("/data/niklas/results/results")
    #results_folder = Path("/data/niklas/results/results/GritLM__GritLM-7B-noinstruct")
    #results_folder = Path("/data/niklas/results/results/GritLM__GritLM-7B-pertasktypeinstruct")
    #results_folder = Path("/data/niklas/results/results/GritLM__GritLM-7B-oneinstruct")
    slurm_jobs_folder = Path(__file__).parent / "slurm_jobs"

    # Select via model name
    # model_names = [
    #     "GritLM/GritLM-7B",
    #     "GritLM/GritLM-8x7B",
    # ]

    # Select tasks
    # tasks = mteb.get_tasks(
    #     #task_types=[
    #     #    "BitextMining",
    #     #    "Classification",
    #     #    "Clustering",
    #     #    "MultilabelClassification",
    #     #    "PairClassification",
    #     #    "Reranking",
    #     #    "Retrieval",
    #     #    "InstructionRetrieval",
    #     #    "STS",
    #     #    "Summarization",
    #     #],
    #     tasks=[
    #         "BornholmBitextMining",
    #         "GreekLegalCodeClassification",
    #     ],
    # )

    # tasks = mteb.get_benchmark("MTEB(eng, classic)")

    # Custom removal of tasks/types
    # tasks_to_remove = [
    #     "ClimateFEVER",
    # ]
    # tasks = [t for t in tasks if t.metadata.name not in tasks_to_remove]
    # tasks = [t for t in tasks if t.metadata.type in ["PairClassification"]]

    slurm_jobs_folder.mkdir(exist_ok=True)

    import json
    with open("/data/niklas/results/missing_results_13.json", "r") as f:
        x = json.load(f)

    slurm_job_files = []
    from mteb.models import MODEL_REGISTRY
    for i, (bench, models) in enumerate(list(x.items())):
        # if bench in ["LongEmbed", "MTEB(Europe, beta)", "MTEB(Indic, beta)", "MTEB(jpn)", "MTEB(multilingual)"]: continue
        print(f"Running benchmark {bench}")
        for j, (model, tasks) in enumerate(models.items()):
            print(f"Running model {model}")
            # if j in [0, 1]: continue
            model_names = [x for x in MODEL_REGISTRY if model == x.split("/")[-1]]
            if len(model_names) == 0:
                print(f"Model {model} not found; trying coarse matching")
                model_names = [x for x in MODEL_REGISTRY if model in x.split("/")[-1]]
                if len(model_names) == 0:
                    print(f"Model {model} still not found; trying coarser matching")
                    model_names = [x for x in MODEL_REGISTRY if model in x]
                    if len(model_names) == 0:
                        print(f"Model {model} not found; skipping")
            if len(model_names) > 1:
                print(f"Model {model} has multiple matches: {model_names}")
                continue
            model_name = model_names[0]
            for k, task_name in enumerate(tasks):
                task = mteb.get_task(task_name)
                if (any([x in model_name for x in ["GritLM", "SFR", "gte-Qwen", "e5-mistral-7b-instruct"]])) or (any([x in task.metadata.name for x in ["MSMARCO"]])):
                    prefix = slurm_prefix_8gpus
                else:
                    prefix = slurm_prefix
                slurm_job_file = create_slurm_job_file(
                    model_name,
                    task.metadata.name,
                    results_folder,
                    prefix,
                    slurm_jobs_folder,
                )
                slurm_job_files.append(slurm_job_file)
                #break
            #break
        # if len(slurm_job_files) > 0:
        #     break

    # slurm_job_files = create_slurm_job_files(
    #     model_names, tasks, results_folder, slurm_prefix, slurm_jobs_folder
    # )

    run_slurm_jobs(slurm_job_files)
