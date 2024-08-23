import os

from steps.features_fusion.fuser import FeaturesFuser
from utils.load_data import load_features_npy_files
from utils.save_data import save_npy_files


def fuse_and_save_features(npy_files_info: dict):
    """
    Fuse and save features.

    :param npy_files_info: dict
    """

    features_fuser_ = FeaturesFuser(npy_files_info)
    fused_npy_files_ = features_fuser_.fuse_features()
    save_npy_files(fused_npy_files_, is_features=True)


def fusion_pipeline(standardized_features_folder: str):
    for folder in os.listdir(standardized_features_folder):
        # Loading .npy files
        folder_path = os.path.join(standardized_features_folder, folder)
        standardized_npy_files = load_features_npy_files(folder_path)
        datasets = os.listdir(folder_path)

        # Fusing and saving features
        if "treino" in datasets:
            fuse_and_save_features(standardized_npy_files)
        else:
            for dataset in datasets:
                fuse_and_save_features(standardized_npy_files)
