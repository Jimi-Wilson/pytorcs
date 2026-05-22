from pathlib import Path
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pytorcs.configs import BaseRunConfig
from pytorcs.utils import build_default_callbacks

def train_algorithm(algorithm_class: type[BaseAlgorithm], env, config: BaseRunConfig, run_dir: Path, resume_path: str = None):
    algo_kwargs = config.algorithm_kwargs.copy()

    if config.callbacks is None:
        config.callbacks = build_default_callbacks(run_dir, config.run_name)

    if config.requires_action_noise:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        algo_kwargs["action_noise"] = action_noise

    print(f"\n[Train] Initializing {algorithm_class.__name__}...")

    if resume_path:
        print(f"[Train] Resuming from checkpoint: {resume_path}")
        load_kwargs = {k: v for k, v in algo_kwargs.items() if k not in ("policy", "policy_kwargs")}
        model = algorithm_class.load(resume_path, env=env, **load_kwargs)
    else:
        model = algorithm_class(env=env, **algo_kwargs)

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=config.callbacks,
        reset_num_timesteps=not resume_path,
        tb_log_name=config.run_name,
    )

    model.save(str(run_dir / f"{config.run_name}_final"))