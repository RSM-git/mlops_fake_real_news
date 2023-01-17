from src.utils import load_yaml


def test_load_yaml():
    path = "configs/train_cpu.yaml"
    config = load_yaml(path)
    assert isinstance(config, dict)
