# Experiments

Each subfolder is a self-contained experiment. To run one, point `train_ppo.py` at its `config.py`.

```
experiments/
├── README.md               ← you are here
└── corkscrew_v1/
    ├── config.py           ← experiment definition
    ├── reward.py           ← reward function
    ├── checkpoints/        ← periodic model snapshots (created at runtime)
    ├── best_models/        ← best-lap model + completed lap data (created at runtime)
    ├── tensorboard/        ← TensorBoard logs (created at runtime)
    └── monitor/            ← SB3 Monitor CSVs (created at runtime)
```

---

## Running an experiment

All commands are run from the `gym_torcs/` directory.

```bash
# Start a new training run
python train_ppo.py --config experiments/corkscrew_v1/config-1.py

# Resume from the latest checkpoint
python train_ppo.py --config experiments/corkscrew_v1/config-1.py --resume

# Evaluate the best saved model (5 episodes by default)
python train_ppo.py --config experiments/corkscrew_v1/config-1.py --eval

# Evaluate for more episodes
python train_ppo.py --config experiments/corkscrew_v1/config-1.py --eval --episodes 10
```

---

## Creating a new experiment

1. **Copy an existing folder** (or create a new one):
   ```
   experiments/
   └── my_experiment/
       ├── config.py
       └── reward.py
   ```

2. **Write your reward function** in `reward.py`:
   ```python
   # experiments/my_experiment/reward.py
   import math

   def my_reward(previous, current) -> float:
       progress = ((current.distRaced - previous.distRaced) / 10.0) * math.cos(current.angle)
       off_track = abs(current.trackPos) > 1.2
       return float(progress - (200.0 if off_track else 0.0))
   ```

3. **Write your config** in `config.py`:
   ```python
   # experiments/my_experiment/config-1.py
   from reward import my_reward
   from train_ppo import RunConfig

   cfg = RunConfig(
       reward_fn = my_reward,
   )
   ```
   That's the minimum. Everything else uses sensible defaults.

4. **Run it:**
   ```bash
   python train_ppo.py --config experiments/my_experiment/config-1.py
   ```

---

## RunConfig reference

All fields are optional except `reward_fn`.

| Field | Default | Description |
|---|---|---|
| `reward_fn` | **required** | `(previous_sensors, current_sensors) -> float` |
| `total_timesteps` | `1_000_000` | Total PPO environment steps |
| `ppo_overrides` | `{}` | Keys merged on top of the default PPO kwargs (see below) |
| `env_overrides` | `{}` | Keys merged on top of the default env kwargs (see below) |
| `bc_kwargs` | `None` | BC pre-training settings. `None` = skip BC |
| `num_envs` | `1` | Parallel environments. `>1` requires `use_docker=True` |
| `use_docker` | `False` | Enables Docker container lifecycle management |
| `base_port` | `3001` | First UDP port. Docker uses `base_port + 0 … base_port + (num_envs - 1)` |
| `callbacks` | `None` | Override the default callback list. `None` = use defaults |

### Default PPO kwargs

These are the values `ppo_overrides` merges into. Only specify what you want to change.

```python
DEFAULT_PPO_KWARGS = {
    "policy":        "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps":       2048,
    "batch_size":    256,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "ent_coef":      0.005,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [256, 256], "activation_fn": nn.Tanh},
}
```

### Default env kwargs

```python
DEFAULT_ENV_KWARGS = {
    "sensor_features": ["speedX", "angle", "trackPos", "track"],
    "truncate_limit":  10_000,
}
```

---

## Common config patterns

### Change only the learning rate
```python
cfg = RunConfig(
    reward_fn     = my_reward,
    ppo_overrides = {"learning_rate": 1e-4},
)
```

### Enable BC pre-training with a teacher policy
```python
from pid_racer import pid_policy

cfg = RunConfig(
    reward_fn = my_reward,
    bc_kwargs = {
        "teacher_policy":      pid_policy,   # any callable: (obs) -> action
        "collection_episodes": 30,
        "only_completed_laps": True,
        "epochs":              20,
        "batch_size":          256,
        "lr":                  1e-3,
    },
)
```

### Use a trained PPO model as the BC teacher
```python
from stable_baselines3 import PPO as _PPO

_teacher = _PPO.load("experiments/corkscrew_v1/best_models/best_lap_111.10s_step676119.zip")

cfg = RunConfig(
    reward_fn = my_reward,
    bc_kwargs = {
        "teacher_policy": lambda obs: _teacher.predict(obs, deterministic=True)[0],
        "collection_episodes": 20,
        "only_completed_laps": True,
        "epochs": 15,
        "batch_size": 256,
        "lr": 1e-3,
    },
)
```

### Multi-container Docker run (8 envs)
```python
cfg = RunConfig(
    reward_fn       = my_reward,
    total_timesteps = 2_000_000,
    num_envs        = 8,
    use_docker      = True,
    base_port       = 3001,        # uses ports 3001–3008
    ppo_overrides   = {
        "n_steps": 512,            # 8 envs × 512 = 4096 steps per update
    },
)
```

### Custom callbacks
```python
from stable_baselines3.common.callbacks import EvalCallback

cfg = RunConfig(
    reward_fn = my_reward,
    callbacks = [my_custom_callback],   # replaces the default list entirely
)
```

---

## Tips

- **Folder name = run name.** The experiment folder name is used as the TensorBoard run name and checkpoint prefix. Name it clearly (e.g. `corkscrew_v2_higher_lr`).
- **Outputs are self-contained.** Delete the folder and everything for that experiment is gone cleanly.
- **BC skips on resume.** Running `--resume` always skips BC — the model weights are already warm-started.
- **`n_steps` scales with `num_envs`.** With 8 envs, `n_steps=2048` means 16 384 steps per update. Consider halving `n_steps` when doubling envs.
