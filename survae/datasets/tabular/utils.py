from pyprojroot import here
from pathlib import Path

# spyder up to find the root
# root = here(project_files=[""])


import os
import importlib


def get_survae_path():
    init_path = importlib.util.find_spec("survae").origin
    path = os.path.dirname(os.path.dirname(init_path))
    return path


# def get_data_path_file():
#     path = get_survae_path()
#     file = os.path.join(path, 'data_path')
#     return file


DATA_PATH = Path(get_survae_path()).joinpath("data")

# makedir
DATA_PATH.mkdir(parents=True, exist_ok=True)


def get_data_path():
    return str(DATA_PATH)
