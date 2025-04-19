import os
import signal
from utils import init_logger

logger = init_logger(rank=0)

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID")
SHOULD_QUIT = False

def handle_sigusr1(signum, frame):
    logger.info(f"Received signal {signum}. Quitting soon...")
    global SHOULD_QUIT
    
def is_time_up():
    global SHOULD_QUIT
    return SHOULD_QUIT

signal.signal(signal.SIGUSR1, handle_sigusr1)
signal.signal(signal.SIGTERM, handle_sigusr1)
