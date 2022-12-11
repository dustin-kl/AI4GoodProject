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
        "epochs": 3,
    },
    "baseline": {
        "n_input_channels": 4,
        "batch_size": 8,
        "epochs": 3,
    },
    "transunet": {
        "batch_size": 2,
        "patches": {
            "size": (16, 16),
            "grid": (16, 16),
        },
        "hidden_size": 768,
        "transformer": {
            "mlp_dim": 3072,
            "num_heads": 12,
            "num_layers": 12,
            "attention_dropout_rate": 0.0,
            "dropout_rate": 0.5,
        },
        "classifier": "seg",
        "representation_size": None,
        "resnet_pretrained_path": None,
        "pretrained_path": None,
        "patch_size": 16,
        "decoder_channels": (256, 128, 64, 16),
        "resnet": {
            "num_layers": (3, 4, 9),
            "width_factor": 1,
        },
        "skip_channels": [512, 256, 64, 16],
        "n_classes": 3,
        "n_skip": 3,
        "activation": "softmax",
        "calssifier": "seg",
    }
}
