import os
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.utils import Generic

def main():
    directory = Generic.get_directory(__file__)
    data = os.listdir(directory)
    data = shuffle(data)
    train, test = train_test_split(data, test_size=0.2)
    validation, test = train_test_split(test, test_size=0.5)

    with open(directory + "data_split.json", "w") as f:
        json.dump({"train": train, "validation": validation, "test": test}, f)


if __name__ == "__main__":
    main()