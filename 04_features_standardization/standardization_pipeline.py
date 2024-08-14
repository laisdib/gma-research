import os

from standardizer import FeaturesStandardizer
from utils.load_data import load_features_npy_files
from utils.save_data import save_npy_files


def standardize_and_save_features(npy_files_info: dict):
    """
    Standardize and save features.

    :param npy_files_info: dict
    """

    features_standardizer_ = FeaturesStandardizer(npy_files_info)
    standardized_npy_files_ = features_standardizer_.standardize_features()
    save_npy_files(standardized_npy_files_, is_features=True)


root_path = "../data/features"
non_standardized_features_folder = f"{root_path}/non_standardized"
standardized_features_folder = f"{root_path}/standardized"

for folder in os.listdir(non_standardized_features_folder):
    # Loading .npy files
    folder_path = os.path.join(non_standardized_features_folder, folder)
    non_standardized_npy_files = load_features_npy_files(folder_path)
    datasets = os.listdir(folder_path)

    # Standardizing and saving features
    if "treino" in datasets:
        standardize_and_save_features(non_standardized_npy_files)
    else:
        for dataset in datasets:
            standardize_and_save_features(non_standardized_npy_files)
