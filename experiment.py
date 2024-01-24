import logging
import sys
import traceback
from datetime import datetime
from logging import exception, info, warning
from pathlib import Path
from typing import Optional

from pytz import timezone

import constants as c


class Experiment:
    def __init__(
        self,
        name: Optional[str] = None,
        directory: Optional[Path] = None,
        seed: Optional[int] = None,
        dry_run: bool = False,
    ):
        if dry_run:
            warning("NOTE: Experiment is set up as dry run!")
        self.name = (
            name
            if name is not None
            else datetime.now(timezone("Europe/Berlin")).strftime("%Y-%m-%d_%H:%M")
        )
        parent_dir = directory if directory is not None else c.experiments_dir
        self.working_dir = parent_dir / self.name
        self.seed = seed
        self.dry_run = dry_run
        if self.seed is not None:
            from pytorch_lightning.utilities.seed import seed_everything

            seed_everything(self.seed, workers=True)
        if not self.dry_run:
            self.working_dir.mkdir(exist_ok=name is not None)
            log_file_path = self.working_dir / "experiment.log"
            self._std_redirect = open(log_file_path, "a")
            self._log_handler = logging.FileHandler(log_file_path)
            self._log_handler.setLevel(logging.DEBUG)
            self._log_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s : %(levelname)s : %(message)s", "%Y-%m-%d %H:%M:%S"
                )
            )

    def __enter__(self):
        if not self.dry_run:
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = self._std_redirect
            sys.stderr = self._std_redirect
            logging.getLogger().addHandler(self._log_handler)
        info(f"Starting experiment {self.name}.")
        self._start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        # Try to mitigate some memory leaks experienced during the stain normalization experiments.
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

        duration = datetime.now() - self._start_time
        if exc_type is not None:
            if not self.dry_run:
                with open(self.working_dir / "error.log", "a") as f:
                    print(
                        f"Experiment {self.name} failed. Details:\n{traceback.format_exc()}.",
                        file=f,
                    )
            exception(f"Experiment {self.name} failed after {duration}.")
        else:
            info(f"Finished experiment {self.name}. Took {duration}.")
        if not self.dry_run:
            self._std_redirect.flush()
            self._std_redirect.close()
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            logging.getLogger().removeHandler(self._log_handler)
        return None
