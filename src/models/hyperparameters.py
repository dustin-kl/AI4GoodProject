
hyper_parameters = {
    "unet": {
        "n_channels": 4,
        "batch_size": 3,
        "epochs": 40,
        "n_classes": 3,
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
            "num_layers": 5,
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
    },
    "baseline": None,
    "attention": None,
}
