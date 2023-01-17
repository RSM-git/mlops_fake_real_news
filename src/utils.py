import yaml
from yaml.loader import SafeLoader


def load_yaml(file_path: str):
    with open(file_path, "r") as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data
