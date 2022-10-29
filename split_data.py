from parser import args
from src.utils import Generic

def main():
    directory = Generic.get_directory(__file__)
    data_list = Generic.list_files(directory + "/dataset")
    data_train, data_test = Generic.split_data(data_list)
    data_split = {
        "train": data_train,
        "test": data_test,
    }
    if (args["test"]):
        Generic.to_json(data_split, directory + "/data_split_test.json")
    else:
        Generic.to_json(data_split, directory + "/data_split.json")


if __name__ == "__main__":
    main()