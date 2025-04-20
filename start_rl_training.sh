#!/usr/bin/env bash

# -----------------------------------------------------------
# Trap Setup: Catch signals like Ctrl+C and kill old servers.
# -----------------------------------------------------------
cleanup() {
    echo "[start_rl_training.sh] Caught a termination signal. Stopping servers..."
    ./stop_vllm_server.sh
    exit 130
}

# Trap Ctrl+C (SIGINT) and SIGTERM
trap cleanup INT TERM

ACCELERATE_CONFIG=""
SERVER_ARGS=()
TRAIN_ARGS=()

# -----------------------------------------------------------
# Parse all arguments
# -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config_file)
            # This argument is for accelerate only
            ACCELERATE_CONFIG="$2"
            shift 2
            ;;
        *)
            # Everything else goes to both server and train
            SERVER_ARGS+=("$1")
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

# -----------------------------------------------------------
# Kill any old server processes, wait briefly
# -----------------------------------------------------------
./stop_vllm_server.sh
sleep 5

# -----------------------------------------------------------
# Start VLLM server in background
# -----------------------------------------------------------
echo "[start_rl_training.sh] Starting VLLM server in the background..."
./start_vllm_server.sh "${SERVER_ARGS[@]}" &
SERVER_PID=$!

echo "[start_rl_training.sh] Waiting for VLLM server to become available..."

# -----------------------------------------------------------
# Poll the server's docs endpoint
# -----------------------------------------------------------
until curl -sSf http://localhost:8005/docs > /dev/null 2>&1
do
    echo "[start_rl_training.sh] Server not up yet. Retrying in 30s..."
    sleep 30
done

echo "[start_rl_training.sh] VLLM server is up! Proceeding with Accelerate launch..."

# -----------------------------------------------------------
# Run training
# -----------------------------------------------------------
accelerate launch --config_file "$ACCELERATE_CONFIG" train_rl.py "${TRAIN_ARGS[@]}"

# -----------------------------------------------------------
# Kill servers after training
# -----------------------------------------------------------
./stop_vllm_server.sh

echo "[start_rl_training.sh] Done."
