from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class TerminalLoggingCallback(BaseCallback):
    """Monitors vector workers from the main process thread and displays clean, 
    scannable summaries whenever an environment finishes an episode."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_counts = {}
        self.best_lap = float("inf")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            if env_idx not in self.episode_counts:
                self.episode_counts[env_idx] = 0

            if done:
                self.episode_counts[env_idx] += 1
                ep_num = self.episode_counts[env_idx]

                # Extract telemetry from the info dictionary
                dist = info.get("dist_raced", 0.0)
                status = info.get("status_flag", "UNKNOWN")
                lap_time = info.get("lap_time", 0.0)

                # Select a status marker with simple clean text design
                if status == "LAP_COMPLETE":
                    status_str = f"🏁 LAP COMPLETE ({lap_time:.2f}s)"
                    if lap_time < self.best_lap:
                        self.best_lap = lap_time
                elif status == "OFF_TRACK":
                    status_str = "✗ OFF TRACK"
                elif status == "ANTI_STALL":
                    status_str = "🛑 ANTI-STALL"
                else:
                    status_str = "⏱ TRUNCATED"

                # Output an aligned layout line containing critical metrics for marking
                print(
                    f"[Worker {env_idx}]  "
                    f"Episode {ep_num:<3} | "
                    f"Steps: {self.num_timesteps:<6,} | "
                    f"Distance: {dist:>5.0f}m | "
                    f"Result: {status_str}"
                )
        return True


class LapTimeCallback(BaseCallback):
    """Logs every completed lap time to TensorBoard and stdout."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.best_lap = float("inf")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            lt = info.get("lap_time", 0)
            if lt and lt > 0:
                self.logger.record("custom/lap_time", lt)
                if lt < self.best_lap:
                    self.best_lap = lt
                    self.logger.record("custom/best_lap_time", lt)
                    print(f"\n  🏁 New best lap: {lt:.2f}s  (step {self.num_timesteps})")
        return True


class BestLapSaveCallback(BaseCallback):
    """
    Saves the model whenever a new best lap time is recorded.

    Multi-env safe: inspects info dicts from all envs each step without
    maintaining a shared episode buffer (which would mix trajectories
    across envs and corrupt the data).
    """

    def __init__(self, save_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.best_lap = float("inf")

    def _on_training_start(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        for done, info in zip(
                self.locals.get("dones", [False]),
                self.locals.get("infos", [{}]),
        ):
            if not done:
                continue

            lt = info.get("lap_time", 0)
            if lt and lt > 0 and lt < self.best_lap:
                self.best_lap = lt
                model_path = self.save_dir / f"best_lap_{lt:.2f}s_step{self.num_timesteps}"
                self.model.save(str(model_path))
                print(f"\n  🏆 New best lap: {lt:.2f}s → {model_path.name}.zip")

        return True
