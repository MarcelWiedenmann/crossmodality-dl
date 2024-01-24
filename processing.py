import abc
import json
import traceback
from datetime import datetime
from logging import exception, info, warning
from pathlib import Path
from pprint import pformat
from typing import Any, Optional


class ProcessingConfig(abc.ABC):
    def to_file(self, path: Path) -> None:
        with open(path, "w") as f:
            return json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def from_file(
        cfg_class: "Type[ProcessingConfig]", path: Path
    ) -> "ProcessingConfig":
        with open(path, "r") as f:
            return cfg_class.from_dict(json.load(f))


class Processing(abc.ABC):
    def __init__(
        self,
        name: str,
        working_dir: Path,
        cfg: Optional[ProcessingConfig],
        dry_run: bool,
    ):
        self._name = name
        self._working_dir = working_dir
        self._cfg = cfg
        self._dry_run = dry_run
        cfg_str = (
            pformat(dict(sorted(self._cfg.to_dict().items())))
            if self._cfg is not None
            else "no configuration"
        )
        info(f"Setting up {self._name} with configuration:\n{cfg_str}.")
        if not self._dry_run and self._cfg is not None:
            self._cfg.to_file(self._working_dir / (self._name + ".json"))

    @abc.abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def run(self, *args, **kwargs) -> Any:
        try:
            if self._dry_run:
                warning("NOTE: The following is a dry run!")
            info(f"Starting {self._name}.")
            start_time = datetime.now()
            output = self._run(*args, **kwargs)
            info(f"Finished {self._name}. Took {datetime.now() - start_time}.")
            return output
        except BaseException as ex:
            if not self._dry_run:
                with open(self._working_dir / "error.log", "a") as f:
                    print(
                        f"{self._name} failed. Details:\n{traceback.format_exc()}.",
                        file=f,
                    )
            exception(ex)
            raise ex
