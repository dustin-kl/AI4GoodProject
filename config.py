config = dict()

config["height"] = 768
config["width"] = 1152

config["dataset"] = "EEG"
config["batch_size"] = 32
config["shuffle"] = True

config["feature_list"] = ["TMQ", "U850", "V850", "PRECT"]  # only use one gpu
