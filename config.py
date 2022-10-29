import json

from parser import args
from src.utils import Generic

"""
In this file we configure which model to run and what to to with it
We also select the dataset which we want to operate on
"""
config = dict()

config["height"] = 768
config["width"] = 1152

"""
Select which model to run 

Available modes: 'train', 'test', 'inference' 
"""
config["model"] = "CNN"  # select which model to run
config["mode"] = "train"


"""
Dataset related settings 
"""
config["dataset"] = "EEG"
config["batch_size"] = 32
config["shuffle"] = True

if (args["test"]):
    with open(Generic.get_directory(__file__) + "/data_split_test.json") as f:
        data_split = json.load(f)
        config["train_dataset"] = data_split["train"]
        config["test_dataset"] = data_split["test"]
else:
    with open(Generic.get_directory(__file__) + "/data_split.json") as f:
        data_split = json.load(f)
        config["train_dataset"] = data_split["train"]
        config["test_dataset"] = data_split["test"]
    
"""
Feature List
"""
config["features"] = ["T500"]

"""
Training related settings
"""
config["gpu_ids"] = [0]  # only use one gpu

"""
Evaluation related settings 
"""
# TODO:implement
