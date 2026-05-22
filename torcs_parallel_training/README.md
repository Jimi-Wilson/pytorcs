# TORCS Training Environment

## System Compatibility

* Fedora 44 (Tested) 
* Windows 11 (Tested)
* MacOs (Untested)
* WSL2 (Tested)


## Prerequisites

Ensure you have the following system:
* [Docker](https://docs.docker.com/get-docker/)
* [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package installer and resolver)


## Setup

1. **Sync dependencies**
    ```bash
    uv sync
    ```
    This reads the `pyproject.toml` and locks a virtual environment with required libraries like `gymnasium`, `stable-baselines3`, and `tensorboard`.

## Training a model

Training a model is done by running the `train.py` script.
The most basic training command is: 
```bash
uv run python train.py --config path/to/config.py
```

`train.py` argument:
1. `--config` - Path to the config file. (Required)
2. `--resume` - Path to the model to resume training from.

### Config File
Training config files define the PPO models training parameters and the environment training parameters.

To make a config file, take a look at the `example_config.py` file in the examples' directory.

### Reward Function

Reward functions calculate the reward of the agent based on the current and previous state of the environment.

[List of sensors and actuators for reward functions (See Page 13 & 14)](https://arxiv.org/pdf/1304.1672)


## Evaluating a model
Evaluation is done by running the `evaluate.py` script.
The most basic evaluation command is: 
```bash
uv run python evaluate.py --config path/to/config.py --model --path/to/model.zip
```

### Visual Mode
You can visualize the model's performance by adding the --visual flag.
This is only known to work on Linux.
```bash
uv run python evaluate.py --config path/to/config.py --model --path/to/model.zip --visual
```

You may have to allow the docker container to access your X11 server.
To do this.
```bash
xhost +local:root
```
