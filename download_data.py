from zipfile import ZipFile

import gdown

from src.utils import get_directory, remove_file


def download_folder(url, output):
    gdown.download(url, output=output)


def unzip(zip_file, destination):
    with ZipFile(zip_file, "r") as zip:
        zip.extractall(destination)


def main():
    directory = get_directory(__file__)
    zip_file = directory + "/Dataset.zip"

    url = "https://drive.google.com/uc?id=11mn7JE1aCGSOPt7y719swNoaDTJSJ3ze"
    download_folder(url, zip_file)

    #unzip(zip_file, directory)

    #remove_file(zip_file)


if __name__ == "__main__":
    main()
