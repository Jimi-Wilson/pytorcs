import importlib
import sys
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from pytorcs.callbacks import LapTimeCallback, BestLapSaveCallback, TerminalLoggingCallback
from pytorcs.configs import BaseRunConfig
from pytorcs.wrappers import FrameSkipWrapper


def get_orchestrator():
    root_dir = Path(__file__).resolve().parents[2]
    docker_path = str(root_dir / "docker")

    if docker_path not in sys.path:
        sys.path.insert(0, docker_path)

    import orchestrate
    return orchestrate


def get_timestamped_run_dir(exp_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    return exp_dir / f"{run_name}_{timestamp}"

def kill_torcs_instance(port: int) -> None:
    orchestrate = get_orchestrator()

    orchestrate.kill_torcs_on(port)


def load_config(config_path: str) -> tuple[BaseRunConfig, Path]:
    config_path = Path(config_path).resolve()

    if str(config_path.parent) not in sys.path:
        sys.path.insert(0, str(config_path.parent))

    spec = importlib.util.spec_from_file_location("experiment_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.config, config_path.parent


def make_env(port: int, config, env_kwargs: dict, monitor_dir: Path):
    from pytorcs.torcs_env import TorcsEnv

    def _init():
        skip = env_kwargs.get("frame_skip", 1)
        torcs_kwargs = {k: v for k, v in env_kwargs.items() if k != "frame_skip"}
        env = TorcsEnv(**torcs_kwargs, reward_function=config.reward_function, port=port)
        return Monitor(FrameSkipWrapper(env, skip=skip) if skip > 1 else env, str(monitor_dir))
    return _init


def build_default_callbacks(run_dir: Path, run_name: str) -> list:
    checkpoint_cb = CheckpointCallback(
        save_freq=250_000, save_path=str(run_dir / "checkpoints"), name_prefix=run_name
    )
    return [
        checkpoint_cb,
        LapTimeCallback(verbose=1),
        BestLapSaveCallback(save_dir=run_dir / "best_models", verbose=1),
        TerminalLoggingCallback()
    ]
