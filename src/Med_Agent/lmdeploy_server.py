import os
import subprocess

MODEL_PATH = "MediScope-model-merged"

def start_server():

    cmd = f"""
    lmdeploy serve api_server \
        {MODEL_PATH} \
        --model-name intern-s1-mini \
        --backend turbomind \
        --reasoning-parser intern-s1 \
        --tool-call-parser intern-s1 \
        --cache-max-entry-count 0.1 \
        --max-batch-size 8 \
        --session-len 8192 \
        --server-port 23333
    """

    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    start_server()