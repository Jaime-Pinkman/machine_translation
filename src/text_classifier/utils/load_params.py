import yaml
from text_classifier.config.config import default_params


def load_params(config_file=None):
    if config_file is None:
        return default_params
    else:
        with open(config_file, "r") as file:
            params = yaml.safe_load(file)
        return {
            "learning_rate": float(params["learning_rate"]),
            "batch_size": int(params["batch_size"]),
            "num_epochs": int(params["num_epochs"]),
        }
