import os

def create_dir(path_to_dir):
    ''' check wether the directory already exists. if not, create it '''
    exit = 0       # exit code

    if not os.path.isdir(path_to_dir):
        os.makedirs(path_to_dir)
        exit = 0
    else:
        exit = 1

    return exit

