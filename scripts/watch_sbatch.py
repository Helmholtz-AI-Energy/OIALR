import re
import subprocess
import time
from pathlib import Path


def submit_and_catch():
    output = subprocess.run(["sbatch", "scripts/launch_run.sbatch"], stdout=subprocess.PIPE)
    out_string = output.stdout.decode("utf-8")
    print(out_string[:-1])
    jobid = re.findall(r"\d+", out_string)[0]
    log = Path("/hkfs/work/workspace/scratch/qv2382-madonna/madonna/logs/slurm/")
    log = log / f"slurm-{jobid}"
    print(f"waiting for log -> {log}")
    while not log.is_file():
        time.sleep(1)
    # once its a file, sleep for 2 seconds, then grab the first lines of the file
    time.sleep(5)
    try:
        with open(log) as f:
            lines = [next(f) for _ in range(5)]
    except StopIteration:
        # the job will take a few seconds to load things and will not print anything in the first seconds
        # i.e. there will not be 5 lines in the log 2s after starting the job
        return False, jobid
    kill = False
    for line in lines:
        if "slurmstepd" in line:
            kill = True
            break
    return kill, jobid


if __name__ == "__main__":
    kill = False
    while True:
        kill, jobid = submit_and_catch()
        if not kill:
            print("Successfully launched job")
            exit(0)
        if kill:
            _ = subprocess.run(["scancel", str(jobid)])
