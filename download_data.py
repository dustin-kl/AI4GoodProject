from zipfile import ZipFile

import gdown

from src.utils import Generic


def download_folder(url, output):
    gdown.download(url, output=output)


def unzip(zip_file, destination):
    with ZipFile(zip_file, "r") as zip:
        zip.extractall(destination)


def main():
    directory = Generic.get_directory(__file__)
    zip_file = directory + "/dataset.zip"

    #url = "https://drive.google.com/uc?id=11mn7JE1aCGSOPt7y719swNoaDTJSJ3ze"
    url = "https://drive.google.com/uc?id=12nuEC3O1lfDrvJGIb87A5FyQqNfHZjwY"
    download_folder(url, zip_file)

    #unzip(zip_file, directory)

    #remove_file(zip_file)


if __name__ == "__main__":
    main()
