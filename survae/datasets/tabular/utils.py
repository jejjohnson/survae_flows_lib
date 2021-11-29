from pyprojroot import here
from pathlib import Path

# spyder up to find the root
root = here(project_files=[".home"])

DATA_PATH = Path(root).joinpath("data")

# makedir
DATA_PATH.mkdir(parents=True, exist_ok=True)


def get_data_path():
    return str(DATA_PATH)
