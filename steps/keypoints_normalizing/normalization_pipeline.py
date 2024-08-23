import os

from steps.keypoints_normalizing.normalizer import KeyPointsNormalizer
from utils.load_data import load_npy_files_per_folder
from utils.utils import get_key_points_value, define_key_points_type
from utils.save_data import save_npy_files
from consts.key_points_info import KEY_POINTS_VALUES


def normalization_pipeline(original_key_points_folder: str):
    for folder in os.listdir(original_key_points_folder):
        # Loading .npy files
        folder_path = os.path.join(original_key_points_folder, folder)
        preprocessed_npy_files = load_npy_files_per_folder(folder_path)

        # Defining key-points value and type
        key_points_value = get_key_points_value(KEY_POINTS_VALUES, folder)
        key_points_type = define_key_points_type(KEY_POINTS_VALUES, folder)

        # Normalizing and saving preprocessed key-points
        key_points_normalizer = KeyPointsNormalizer(key_points_type)

        normalized_npy_files = key_points_normalizer.normalize_key_points(preprocessed_npy_files, key_points_value)
        save_npy_files(normalized_npy_files)
