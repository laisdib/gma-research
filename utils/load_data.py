import os
import numpy as np

from pathlib import Path


def list_all_folders_and_subfolders_path(root_dir: str):
    """
    List all folders and subfolders path in a root directory.

    :param root_dir: str
    :return: list
    """

    root_path = Path(root_dir)
    dirs_with_files = set()

    for file in root_path.rglob('*'):
        if file.is_file():
            relative_path = file.parent.relative_to(root_path)
            dirs_with_files.add(str(relative_path))

    return sorted(dirs_with_files)


def load_npy_files_per_folder(root_folder_path: str):
    """
    Load npy files per folder.

    :param root_folder_path: str
    :return: dict
    """

    subfolders_path = list_all_folders_and_subfolders_path(root_folder_path)
    npy_files_per_folder = {"root": root_folder_path}

    for i, folder_name in enumerate(subfolders_path):
        complete_folder_path = os.path.join(root_folder_path, folder_name)
        file_names = [file for file in os.listdir(complete_folder_path) if file.endswith('.npy')]
        npy_files = [os.path.join(complete_folder_path, file_name) for file_name in file_names]

        npy_files_per_folder.update(
            {
                i: {
                    "path": folder_name,
                    "content": []
                }
            }
        )

        for file_name, file in zip(file_names, npy_files):
            try:
                npy_files_per_folder[i]["content"].append(
                    {
                        "file_name": file_name,
                        "content": np.load(file),
                        "shape": np.load(file).shape
                    }
                )
            except Exception:
                npy_files_per_folder[i]["content"].append(
                    {
                        "file_name": file_name,
                        "content": np.load(file, allow_pickle=True),
                        "shape": np.load(file, allow_pickle=True).shape
                    }
                )

    return npy_files_per_folder


def load_features_npy_files(root_folder_path: str):
    """
    Load features npy files.

    :param root_folder_path: str
    :return: dict
    """

    subfolders_path = list_all_folders_and_subfolders_path(root_folder_path)
    subfolders_path = list({os.path.dirname(folder) for folder in subfolders_path})

    npy_files_per_folder = {"root": root_folder_path}

    for i, folder_path in enumerate(subfolders_path):
        npy_files_per_folder.update(
            {
                i: {
                    "path": folder_path,
                    "content": []
                }
            }
        )

        path = os.path.join(root_folder_path, folder_path)
        videos = os.listdir(path)

        for video in videos:
            path_ = os.path.join(path, video)
            file_names = [file for file in os.listdir(path_) if file.endswith('.npy')]
            npy_files = [os.path.join(path_, file_name) for file_name in file_names]

            features = {file_name: np.load(file) for file_name, file in zip(file_names, npy_files)}

            npy_files_per_folder[i]["content"].append(
                {
                    "file_name": video,
                    "features": features
                }
            )

    return npy_files_per_folder
