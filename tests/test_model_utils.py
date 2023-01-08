from src.models.utils import *
import os


def test_get_model():
    model_dir = "models/pretrained_models"
    model_type = "albert-base-v2"
    model = get_model(model_type, model_dir=model_dir)
    assert os.path.exists(os.path.join(model_dir, model_type))
