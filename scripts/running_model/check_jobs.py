import subprocess
from collections import defaultdict

# Function to retrieve job IDs and commands from running jobs
def get_running_jobs():
    try:
        # Get all running jobs for the user
        result = subprocess.run(['squeue', '--user=niklas', '--format=%A'], stdout=subprocess.PIPE, text=True)
        job_ids = result.stdout.strip().split("\n")[1:]  # Skip header

        job_commands = {}

        for job_id in job_ids:
            job_id = job_id.strip()
            if not job_id:
                continue

            # Get detailed info about the job using scontrol
            job_info = subprocess.run(['scontrol', 'show', 'jobid', job_id], stdout=subprocess.PIPE, text=True)
            for line in job_info.stdout.splitlines():
                if line.startswith("   Command="):
                    script_path = line.split("=", 1)[1].strip()
                    job_commands[job_id] = script_path
                    break

        return job_commands
    except Exception as e:
        print(f"Error while fetching job details: {e}")
        return {}

# Function to identify duplicate scripts
def find_duplicate_jobs(job_commands):
    script_to_jobs = defaultdict(list)

    for job_id, script_path in job_commands.items():
        if "mteb" in script_path:
            script_to_jobs[script_path].append(job_id)

    duplicates = {script: jobs for script, jobs in script_to_jobs.items() if len(jobs) > 1}
    return script_to_jobs.keys(), duplicates

# Function to cancel the most recent jobs among duplicates
def cancel_recent_jobs(duplicates):
    for script, jobs in duplicates.items():
        # Sort jobs by ID (oldest first)
        sorted_jobs = sorted(jobs, key=int)
        # Keep the oldest job and cancel the rest
        jobs_to_cancel = sorted_jobs[1:]
        for job_id in jobs_to_cancel:
            try:
                subprocess.run(['scancel', job_id], check=True)
                print(f"Cancelled job {job_id} for script {script}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to cancel job {job_id}: {e}")

if __name__ == "__main__":
    job_commands = get_running_jobs()
    if not job_commands:
        print("No running jobs found.")
        exit()

    all_scripts, duplicates = find_duplicate_jobs(job_commands)

    print("\nAll scripts currently running:")
    for script in all_scripts:
        print(f"  {script}")

    print("\nJobs running duplicate scripts:")
    for script, jobs in duplicates.items():
        print(f"  Script: {script}")
        print(f"    Job IDs: {', '.join(jobs)}")

    cancel_recent_jobs(duplicates)
