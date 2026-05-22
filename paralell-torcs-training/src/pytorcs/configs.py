from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Type
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm


@dataclass(kw_only=True)
class BaseRunConfig(ABC):
    run_name: str = "default_run"
    reward_function: Callable
    total_timesteps: int = 1_000_000
    env_kwargs: dict = field(default_factory=dict)
    num_envs: int = 1
    base_port: int = 3001
    callbacks: Optional[List] = None

    @property
    @abstractmethod
    def algorithm_class(self) -> Type[BaseAlgorithm]:
        pass

    @property
    @abstractmethod
    def algorithm_kwargs(self) -> dict:
        pass

    @property
    def requires_action_noise(self) -> bool:
        return False


@dataclass(kw_only=True)
class PPORunConfig(BaseRunConfig):
    ppo_kwargs: dict = field(default_factory=dict)

    @property
    def algorithm_class(self) -> Type[BaseAlgorithm]:
        return PPO

    @property
    def algorithm_kwargs(self) -> dict:
        return self.ppo_kwargs


@dataclass(kw_only=True)
class SACRunConfig(BaseRunConfig):
    sac_kwargs: dict = field(default_factory=dict)

    @property
    def algorithm_class(self) -> Type[BaseAlgorithm]:
        return SAC

    @property
    def algorithm_kwargs(self) -> dict:
        return self.sac_kwargs


@dataclass(kw_only=True)
class TD3RunConfig(BaseRunConfig):
    td3_kwargs: dict = field(default_factory=dict)

    @property
    def algorithm_class(self) -> Type[BaseAlgorithm]:
        return TD3

    @property
    def algorithm_kwargs(self) -> dict:
        return self.td3_kwargs

    @property
    def requires_action_noise(self) -> bool:
        return True