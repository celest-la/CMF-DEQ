import numpy as np
import h5py
from scipy.io import loadmat
import os
from glob import glob
import random


def normalize_x(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x_norm=(x - x_min) / (x_max - x_min + 1e-8)
    return x_norm


def normalize_y(y):
    y = y / (np.abs(y).max() + 1e-8)
    y_real = np.real(y)
    y_imag = np.imag(y)
    y_stacked = np.vstack([y_real, y_imag])
    return y_stacked


def create_shuffled_train_test_dataset(h5_filename, test_size=0.2, random_seed=42):
    # Get all mat files
    mat_files = glob('../data/raw_data/CSM_database5/*.mat')

    # Shuffle the files
    random.seed(random_seed)
    random.shuffle(mat_files)

    # Calculate split point
    num_files = len(mat_files)
    num_test = int(num_files * test_size)
    num_train = num_files - num_test

    train_files = mat_files[:num_train]
    test_files = mat_files[num_train:]

    # Create h5 file
    with h5py.File(h5_filename, 'w') as f:
        # Initialize datasets with the correct shape including channel dimension
        # We'll initialize with first file to get the dimensions right, then resize

        # Process first training file to get dimensions
        if train_files:
            mat = loadmat(train_files[0])
            y = mat['CSM']
            x = mat['map_gt']

            x_normalized = normalize_x(x).astype(np.float32)
            y_normalized = normalize_y(y).astype(np.float32)

            # Create datasets with channel dimension
            f.create_dataset('x_train', data=x_normalized[np.newaxis, np.newaxis, ...],
                             maxshape=(None, 1) + x_normalized.shape, dtype=np.float32)
            f.create_dataset('y_train', data=y_normalized[np.newaxis, np.newaxis, ...],
                             maxshape=(None, 1) + y_normalized.shape, dtype=np.float32)

            # Process remaining training files
            for file in train_files[1:]:
                mat = loadmat(file)
                y = mat['CSM']
                x = mat['map_gt']

                x_normalized = normalize_x(x).astype(np.float32)
                y_normalized = normalize_y(y).astype(np.float32)

                # Resize and add new data
                current_size = f['x_train'].shape[0]
                f['x_train'].resize(current_size + 1, axis=0)
                f['x_train'][current_size] = x_normalized[np.newaxis, ...]

                f['y_train'].resize(current_size + 1, axis=0)
                f['y_train'][current_size] = y_normalized[np.newaxis, ...]

        # Process first test file to get dimensions
        if test_files:
            mat = loadmat(test_files[0])
            y = mat['CSM']
            x = mat['map_gt']

            x_normalized = normalize_x(x).astype(np.float32)
            y_normalized = normalize_y(y).astype(np.float32)

            # Create datasets with channel dimension
            f.create_dataset('x_test', data=x_normalized[np.newaxis, np.newaxis, ...],
                             maxshape=(None, 1) + x_normalized.shape, dtype=np.float32)
            f.create_dataset('y_test', data=y_normalized[np.newaxis, np.newaxis, ...],
                             maxshape=(None, 1) + y_normalized.shape, dtype=np.float32)

            # Process remaining test files
            for file in test_files[1:]:
                mat = loadmat(file)
                y = mat['CSM']
                x = mat['map_gt']

                x_normalized = normalize_x(x).astype(np.float32)
                y_normalized = normalize_y(y).astype(np.float32)

                # Resize and add new data
                current_size = f['x_test'].shape[0]
                f['x_test'].resize(current_size + 1, axis=0)
                f['x_test'][current_size] = x_normalized[np.newaxis, ...]

                f['y_test'].resize(current_size + 1, axis=0)
                f['y_test'][current_size] = y_normalized[np.newaxis, ...]

    print(f"Dataset created: {h5_filename}")
    print(f"Training samples: {num_train}")
    print(f"Test samples: {num_test}")

    # Verify the shapes
    with h5py.File(h5_filename, 'r') as f:
        print(f"x_train shape: {f['x_train'].shape}")  # Should be (num_train, 1, dim1, dim2)
        print(f"y_train shape: {f['y_train'].shape}")  # Should be (num_train, 1, dim1, dim2)
        print(f"x_test shape: {f['x_test'].shape}")  # Should be (num_test, 1, dim1, dim2)
        print(f"y_test shape: {f['y_test'].shape}")  # Should be (num_test, 1, dim1, dim2)


# Create the dataset with shuffled train/test split
create_shuffled_train_test_dataset('../data/h5/CsmPower5.h5')