import subprocess
import sys
import time
from pathlib import Path

DEFAULT_NUM_ENVS = 1
DEFAULT_BASE_PORT = 3001

DOCKER_ROOT = Path(__file__).parent.resolve()

def _run(cmd: list[str], check: bool = True, cwd: Path = DOCKER_ROOT, capture: bool = True):
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
        if check and result.returncode != 0:
            print(
                f"[Orchestrate] ERROR running '{' '.join(cmd)}':\n{result.stderr.strip()}",
                file=sys.stderr,
            )
            result.check_returncode()
        return result
    else:
        result = subprocess.run(cmd, cwd=str(cwd))
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result


def launch(num_envs: int = DEFAULT_NUM_ENVS, base_port: int = DEFAULT_BASE_PORT, visual: bool = False):
    target_dir = DOCKER_ROOT / ("visual" if visual else "cluster")
    service_names = [f"torcs_{base_port + i}" for i in range(1 if visual else num_envs)]

    print("\n[Orchestrate] Checking for and clearing container naming conflicts...")
    for name in service_names:
        _run(["docker", "rm", "-f", name], check=False, capture=True)

    if visual:
        print("[Orchestrate] Granting X server permissions for visual mode...")
        _run(["xhost", "+local:root"], check=False, capture=True)

    print("[Orchestrate] Pulling pre-compiled TORCS image from registry...")
    _run(["docker", "compose", "pull"], check=True, cwd=target_dir, capture=False)

    print(f"[Orchestrate] Launching {num_envs} TORCS container(s) (ports {base_port}–{base_port + num_envs - 1}) ...")
    _run(["docker", "compose", "up", "-d"] + service_names, check=True, cwd=target_dir, capture=False)

def stop(num_envs: int = DEFAULT_NUM_ENVS, base_port: int = DEFAULT_BASE_PORT, visual: bool = False):
    target_dir = DOCKER_ROOT / ("visual" if visual else "cluster")
    service_names = [f"torcs_{base_port + i}" for i in range(1 if visual else num_envs)]

    print(f"[Orchestrate] Stopping {num_envs} container(s) (ports {base_port}–{base_port + num_envs - 1}) ...")
    _run(["docker", "compose", "stop"] + service_names, check=False, cwd=target_dir, capture=False)
    print("[Orchestrate] Done.")

def kill_torcs_on(port: int, wait: float = .5):
    container = f"torcs_{port}"
    _run(["docker", "exec", container, "pkill", "torcs"], check=False, capture=True)
    time.sleep(wait)