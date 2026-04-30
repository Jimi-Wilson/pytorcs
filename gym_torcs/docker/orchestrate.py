import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

DOCKER_DIR = Path(__file__).parent
PROJECT_ROOT = DOCKER_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

DEFAULT_NUM_ENVS = int(os.getenv("NUM_ENVS", 2))
DEFAULT_BASE_PORT = int(os.getenv("BASE_PORT", 3001))


def _run(cmd: list[str], check: bool = True, cwd: Path = DOCKER_DIR):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
    if check and result.returncode != 0:
        print(
            f"[orchestrate] ERROR running '{' '.join(cmd)}':\n{result.stderr.strip()}",
            file=sys.stderr,
        )
        result.check_returncode()
    return result


def launch(num_envs: int = DEFAULT_NUM_ENVS, base_port: int = DEFAULT_BASE_PORT):
    service_names = [f"torcs_{base_port + i}" for i in range(num_envs)]

    def _start(service: str) -> None:
        print(f"[orchestrate] Starting {service}...")
        _run(["docker", "compose", "up", "-d", service])
        print(f"[orchestrate] {service} is up.")

    threads = [
        threading.Thread(target=_start, args=(s,), daemon=True)
        for s in service_names
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(
        f"[orchestrate] {num_envs} TORCS container(s) started "
        f"(ports {base_port}–{base_port + num_envs - 1})."
    )


def stop(num_envs: int = DEFAULT_NUM_ENVS, base_port: int = DEFAULT_BASE_PORT):
    service_names = [f"torcs_{base_port + i}" for i in range(num_envs)]
    print(f"[orchestrate] Stopping: {', '.join(service_names)}")
    _run(["docker", "compose", "stop"] + service_names, check=False)
    print("[orchestrate] Done.")


def kill_torcs_on(port: int, wait: float = .5):
    container = f"torcs_{port}"
    _run(["docker", "exec", container, "pkill", "torcs"], check=False)
    time.sleep(wait)
