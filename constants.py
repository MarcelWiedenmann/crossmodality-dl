import os
from pathlib import Path


home_dir = Path(os.environ["HOME"])

scratch_dir = Path(os.environ["SCRATCH"])

experiments_dir = scratch_dir / "experiments"
experiments_dir.mkdir(exist_ok=True)

logs_dir = scratch_dir / "logs"
logs_dir.mkdir(exist_ok=True)


terminal_he_source_dir = Path(
    "<blank>"
)
serial_he_source_dir = Path(
    "<blank>"
)


terminal2serial = {
    "<blank>": "<blank>"
}
serial2terminal = {v: k for k, v in terminal2serial.items()}

seed = 1634278956

publish_dir = Path("<blank>")
publish_project_name = "<blank>"

remote_fs = "<blank>"
