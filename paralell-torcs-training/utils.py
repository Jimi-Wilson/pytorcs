import os
import sys
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


class SaveBestLapCallback(BaseCallback):
    def __init__(self, save_dir: str, target_time: float = float("inf"), verbose: int = 0):
        super().__init__(verbose)
        self.current_best_lap = target_time
        self.save_dir = save_dir
        self.laps_completed = 0

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            lap_time = info.get("lap_time", 0)

            if lap_time > 0:
                self.laps_completed += 1

                if lap_time < self.current_best_lap:
                    self.current_best_lap = lap_time
                    print(f"\nNew best lap time! Lap {self.laps_completed} completed in {self.current_best_lap:.2f}s")

                    model_path = os.path.join(self.save_dir, f"best_lap_{self.current_best_lap:.2f}s")
                    self.model.save(model_path)
                    print(f"Model saved to {model_path}.zip")
                else:
                    print(f"\nLap {self.laps_completed} completed at {lap_time:.2f}s (Target to beat: {self.current_best_lap:.2f}s)")

        return True

def create_run_folder(base_dir="logs"):
    date_str = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(base_dir, date_str)

    os.makedirs(date_dir, exist_ok=True)

    run_id = 1
    while True:
        run_dir = os.path.join(date_dir, f"run_{run_id}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_dir
        run_id += 1


def kill_torcs_instance(port: int) -> None:
    sys.path.insert(0, str(_HERE / "docker"))
    import orchestrate

    orchestrate.kill_torcs_on(port)