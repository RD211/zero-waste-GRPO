import os
import re
import signal
from utils import init_logger
import subprocess
logger = init_logger(rank=0)

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")
SHOULD_QUIT = False

def get_slurm_time_left():
    job_id = os.getenv("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("SLURM_JOB_ID environment variable not found.")

    result = subprocess.check_output(["scontrol", "show", "job", job_id], text=True)

    tm = re.search(r"TimeLimit=(?:(\d+)-)?(\d{2}):(\d{2}):(\d{2})", result)
    if not tm:
        raise RuntimeError("Could not parse TimeLimit from scontrol output")
    days = int(tm.group(1) or 0)
    hours = int(tm.group(2))
    mins = int(tm.group(3))
    secs = int(tm.group(4))
    time_limit_secs = days * 86400 + hours * 3600 + mins * 60 + secs

    rt = re.search(r"RunTime=(?:(\d+)-)?(\d{2}):(\d{2}):(\d{2})", result)
    if not rt:
        raise RuntimeError("Could not parse RunTime from scontrol output")
    run_days = int(rt.group(1) or 0)
    run_hours = int(rt.group(2))
    run_mins = int(rt.group(3))
    run_secs = int(rt.group(4))
    run_time_secs = run_days * 86400 + run_hours * 3600 + run_mins * 60 + run_secs

    return time_limit_secs - run_time_secs


def handle_sigusr1(signum, frame):
    global SHOULD_QUIT
    logger.info(f"Received signal {signum}. Quitting soon...")
    SHOULD_QUIT = True
    
def received_term():
    global SHOULD_QUIT
    return SHOULD_QUIT

def seconds_left():
    global SLURM_JOB_ID
    if not SLURM_JOB_ID:
        return 1000000
    return get_slurm_time_left()
    

signal.signal(signal.SIGUSR1, handle_sigusr1)
