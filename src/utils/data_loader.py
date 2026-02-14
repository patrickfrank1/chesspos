import os
import glob
from typing import Tuple

import numpy as np


def _increment_postfix(path: str, pattern: str) -> str:
    existing_files = glob.glob(os.path.join(path, pattern))
    if not existing_files:
        postfix = "000"
    else:
        # Extract the postfix numbers and find the highest one, assumes test_001.npz
        existing_postfixes = [int(file.split("_")[-1].split(".")[0]) for file in existing_files]
        highest_postfix = max(existing_postfixes)
        postfix = f"{highest_postfix + 1:03d}"
    return postfix


def save_train_test(path: str, train_split: np.ndarray, test_split: np.ndarray) -> None:
    train_path, test_path = f"{path}/train", f"{path}/test"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    train_postfix = _increment_postfix(train_path, "position_tensor_*.npz")
    test_postfix = _increment_postfix(test_path, "position_tensor_*.npz")
    postfix = max(train_postfix, test_postfix)
    np.savez_compressed(f"{train_path}/position_tensor_{postfix}.npz", data=train_split)
    np.savez_compressed(f"{test_path}/position_tensor_{postfix}.npz", data=test_split)


def load_train_test(path: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
    train, test = None, None
    with np.load(f"{path}/train/{filename}.npz") as data:
        train = data["data"]
    with np.load(f"{path}/test/{filename}.npz") as data:
        test = data["data"]
    return train, test
