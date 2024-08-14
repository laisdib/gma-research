import os
import numpy as np
import shutil

from pathlib import Path


def move_all_files_and_folders(src_root: str, dst_root: str, folders_to_move: list):
    """
    Move all file and folders from a path to another one.

    :param src_root: str
    :param dst_root: str
    :param folders_to_move: list
    """

    src_path = Path(src_root)
    dst_path = Path(dst_root)

    for folder_name in folders_to_move:
        folder_path = src_path / folder_name

        if folder_path.exists() and folder_path.is_dir():
            shutil.move(str(folder_path), str(dst_path / folder_name))


def create_folders(folder_path: str):
    """
    Create folders.

    :param folder_path: str
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def save_data(data: np.ndarray, output_path: str, file_name: str):
    """
    Save ndarray in file.

    :param data: np.ndarray
    :param output_path: str
    :param file_name: str
    """

    create_folders(output_path)

    output_path = os.path.join(output_path, file_name)
    np.save(output_path, data)


def save_npy_files(npy_files_info: dict, folder_name: str = None, is_features: bool = False):
    """
    Save npy files.

    :param npy_files_info: dict
    :param folder_name: str
    :param is_features: bool
    """

    for index in list(npy_files_info.keys())[1:]:
        if folder_name is not None:
            file_path = os.path.join(folder_name, npy_files_info[index]["path"])
        else:
            file_path = npy_files_info[index]["path"]

        general_info = npy_files_info[index]["content"]

        for info in general_info:
            file_name = info["file_name"]

            if not is_features:
                key_points = info["content"]

                destination_path = os.path.join(npy_files_info["root"], file_path)
                save_data(key_points, destination_path, file_name)
            else:
                features = info["features"]

                for name, feature in features.items():
                    destination_path = os.path.join(npy_files_info["root"], file_path, file_name)

                    if ".npy" not in name:
                        name = f"{name}.npy"

                    save_data(feature, destination_path, name)
