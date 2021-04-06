import json
import os
import re
import shutil
import config
from tqdm import tqdm


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


if __name__ == '__main__':
    files = getListOfFiles(config.ABz_directory)
    r = re.compile(r"[^/]*\.json")
    for src in tqdm(files):
        dest = config.ABz_directory_aggregated + r.search(src)[0]
        shutil.move(src, dest)

