#!/usr/bin/env python

from os.path import isfile, join, splitext, basename
from os import listdir, walk
from datetime import datetime

OR = lambda z: reduce(lambda x, y: x or y, z, True)

def get_files(path, type=None):
    files = [f for f in listdir(path) if isfile(join(path, f))]

    if not type is None:
        files = filter(lambda f: f.endswith('.'+type), files)

    return files

def get_files_recursive(path, dirs_get=[], type=None, full=True):
    in_dir = lambda path, directory: path.endswith(directory) 
    in_dirs = lambda path, directories: OR([in_dir(path, directory) for directory in directories])
    prepend_path = lambda path, files: [join(path, f) for f in files]   

    if full: 
        files = reduce(lambda x, y: x + y, [prepend_path(r, fs) for r, _, fs in walk(path) if in_dirs(r, dirs_get)])
    else:
        files = reduce(lambda x, y: x + y, [f for r, _, fs in walk(path) if in_dirs(r, dirs_get)])

    if not type is None:
        files = filter(lambda f: f.endswith('.'+type), files)

    return files

def sort_by_timestamp(files, delim='_', format='%Y-%m-%dT%H%M%S'):
    basenames = [splitext(basename(f))[0] for f in files]
    timestamps = [datetime.strptime(re.split('_', base)[-1], format) for base in basenames]
    sorted_idxs = np.argsort(timestamps)
    return list(np.array(files)[sorted_idxs])

