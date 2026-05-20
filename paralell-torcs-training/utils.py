import sys
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from datetime import datetime
from pathlib import Path


def get_timestamped_run_dir(exp_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    return exp_dir / f"{run_name}_{timestamp}"

def kill_torcs_instance(port: int) -> None:
    sys.path.insert(0, str(_HERE / "docker"))
    import orchestrate

    orchestrate.kill_torcs_on(port)