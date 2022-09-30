import os

def get_directory(file):
    return os.path.dirname(os.path.realpath(file))

def remove_file(file):
    return os.remove(file)