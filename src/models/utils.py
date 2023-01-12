import os

import transformers


def get_model(
    model_type: str = None,
    num_labels: int = 2,
    model_dir: str = "models/pretrained_models",
) -> transformers.AutoModelForSequenceClassification:
    """
    Loads a pretrained model for sequence classificadtion either from disk or
    huggingface and saves the pretrained model on the disk
    for later use
    Args:
        model_type: Huggingface model type. Is inferred automatically if
            weights_name is provided
        model_dir: the directory that we expect the model to be storred in
    Returns:
        model: pretrained model
    """
    # load model architecture
    path_pretrained_model = os.path.join(model_dir, model_type)
    if os.path.exists(path_pretrained_model):
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            path_pretrained_model
        )
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_type, num_labels=num_labels
        )
        model.save_pretrained(f"{model_dir}/{model_type}")
    return model
