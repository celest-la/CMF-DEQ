import numpy as np
import h5py
from scipy.io import loadmat
import os
from glob import glob
import random

def normalize_x(x):
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min + 1e-8)

def normalize_y(y):
    y = y / (np.abs(y).max() + 1e-8)
    y_real = np.real(y)
    y_imag = np.imag(y)
    return np.vstack([y_real, y_imag])

def create_split_datasets(mat_files, grp, f, name_x, name_y):
    """
    Crée (ou étend) deux datasets sous le groupe grp :
    grp/name_x et grp/name_y, en y empilant les exemples.
    """
    for i, filepath in enumerate(mat_files):
        mat = loadmat(filepath)
        x = normalize_x(mat['map_gt']).astype(np.float32)
        y = normalize_y(mat['CSM']).astype(np.float32)

        # new axis pour channel puis batch
        x = x[np.newaxis, np.newaxis, ...]
        y = y[np.newaxis, np.newaxis, ...]

        if i == 0:
            # création du dataset au premier exemple
            f.create_dataset(f"{grp}/{name_x}", data=x,
                             maxshape=(None,)+x.shape[1:], dtype=np.float32)
            f.create_dataset(f"{grp}/{name_y}", data=y,
                             maxshape=(None,)+y.shape[1:], dtype=np.float32)
        else:
            # extension + écriture
            dsx = f[f"{grp}/{name_x}"]
            dsy = f[f"{grp}/{name_y}"]
            dsx.resize(dsx.shape[0] + 1, axis=0)
            dsx[-1] = x
            dsy.resize(dsy.shape[0] + 1, axis=0)
            dsy[-1] = y

def create_h5_with_splits(
    h5_filename,
    raw_folder,
    test_size=0.2,
    valid_size=0.1,
    frequency: float = None,
    random_seed=42
):
    """
    - raw_folder : dossier contenant tes .mat
    - test_size  : fraction réservée au test
    - valid_size : fraction (du reste) réservée à la validation
    - frequency  : valeur à stocker dans l'attrs de chaque split
    """
    mat_files = glob(os.path.join(raw_folder, '*.mat'))
    print(f"[DEBUG] raw_folder = {raw_folder}")
    print(f"[DEBUG] mat_files ({len(mat_files)}): {mat_files}")
    random.seed(random_seed)
    random.shuffle(mat_files)

    n_total = len(mat_files)
    n_test  = int(n_total * test_size)
    n_train_valid = n_total - n_test
    n_valid = int(n_train_valid * valid_size)
    n_train = n_train_valid - n_valid

    train_files = mat_files[:n_train]
    valid_files = mat_files[n_train:n_train + n_valid]
    test_files  = mat_files[n_train + n_valid:]

    with h5py.File(h5_filename, 'w') as f:
        # Optionnel : stocke la fréquence au niveau racine
        if frequency is not None:
            f.attrs['frequency'] = frequency

        # Crée chaque split
        for grp, files in [('train', train_files),
                           ('valid', valid_files),
                           ('test',  test_files)]:
            grp_obj = f.create_group(grp)
            # hérite de la fréquence
            if frequency is not None:
                grp_obj.attrs['frequency'] = frequency
            create_split_datasets(files, grp, f, 'x', 'y')

    print(f"=== HDF5 créé : {h5_filename} ===")
    print(f" Train : {len(train_files)}, Valid : {len(valid_files)}, Test : {len(test_files)}")


create_h5_with_splits(
    h5_filename='../data/h5/CsmPower_good_4p2.h5',
    raw_folder='../data/raw_data/CSM_databasecloud_big5000',
    test_size=0.1,
    valid_size=0.1,
    frequency=4.2e6,        # la fréquence de tes données
)