import pickle
from simulator.simulation import main_path
import os
import numpy as np


def dump_files(path, objs):
    make_folder(os.path.dirname(path))
    with open(os.path.join(main_path, path), 'wb') as f:
        pickle.dump(objs, f)


def load_files(path):
    make_folder(os.path.dirname(path))
    with open(os.path.join(main_path, path), 'rb') as f:
        objs = pickle.load(f)
        return objs


def make_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_or_recompute_and_cache(path, fct, replace=False, verbose=False):
    if replace or not os.path.isfile(path):
        if verbose:
            print(path, 'not found, will be recomputed')
        if not callable(fct):
            IOError('fct must be callable')
        files = fct()
        dump_files(path, files)
    else:
        if verbose:
            print(path, 'found, load from there')
        files = load_files(path)

    return files


def list_dir_clean(dir):
    return [file for file in np.sort(os.listdir(dir)) if not file.startswith('.')]
