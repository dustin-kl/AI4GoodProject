"""
Here we can store the hyperparameter configurations for each model
"""

from config import config

params = {
    "CNN": {  # just an example
        "lr": 1e-4,
        "batch_size": 16,
        "epochs": 100,
        "in_channels": 16,
        "kernel_size": 32,
        "nb_filters": 64,
        "depth": 4,
        "stride": 2,
    },
    "other_model": {},
    "model": {
        "batch_size": 32,
        "epochs": 1,
    },
}
