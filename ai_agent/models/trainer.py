# ai_agent/models/trainer.py
import subprocess
import logging
import os


class ModelTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def train_model(self):
        """Train het model."""
        huidige_cwd = os.getcwd()
        run_dir = os.path.dirname(huidige_cwd)
        print(run_dir)
        try:
            subprocess.run(
                [
                    "python",
                    "training.py",
                    self.config.SYMBOL,
                    self.config.DATA_TYPE,
                    self.config.RUN_NR,
                ],
                cwd=run_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            self.logger.info("Model training completed.")
            print("Model training completed.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Model training failed: {e}")
