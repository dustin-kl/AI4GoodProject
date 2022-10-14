import os
import sys
# import torch
from config import config
from src.models.hyperparameters import params
import os
import json

class Logger:
    def __init__(self, print_fp=None):
        self.terminal = sys.stdout
        self.log_file = "out.txt" if print_fp is None else print_fp
        self.encoding = sys.stdout.encoding

        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
            print(f'removed {self.log_file}')

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, "a") as log:
            log.write(message)

    def flush(self):
        pass

def get_directory(file):
    return os.path.dirname(os.path.realpath(file))

def remove_file(file):
    return os.remove(file)