from os.path import join
from zipfile import ZipFile

import gdown

from parser import args
from src.utils import Generic


def download_folder(url, output):
    gdown.download(url, output=output)


def unzip(zip_file, destination):
    with ZipFile(zip_file, "r") as zip:
        zip.extractall(destination)


def main():
    directory = Generic.get_directory(__file__)

    if (args["test"]):
        url = "https://drive.google.com/uc?id=1tcbp4JUbIUyTm9qI9RyjTwRPkKr6Fdlx"
        download_folder(url, join(directory, "dataset-test.zip"))
    else:
        url = "https://drive.google.com/uc?id=12nuEC3O1lfDrvJGIb87A5FyQqNfHZjwY"
        download_folder(url, join(directory, "dataset.zip"))

    

    #unzip(zip_file, directory)

    #remove_file(zip_file)


if __name__ == "__main__":
    main()
