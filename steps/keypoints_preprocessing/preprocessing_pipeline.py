import os

from steps.keypoints_preprocessing.preprocessor import KeyPointsPreProcessor
from utils.save_data import create_folders, move_all_files_and_folders, save_npy_files
from utils.load_data import load_npy_files_per_folder
from utils.utils import get_key_points_value


KEY_POINTS_VALUES = {
    "OpenPose": 18,
    "MediaPipe": 33
}


def save_corrected_data(corrected_npy_files: dict, folder_path: str):
    background_path = os.path.join(corrected_npy_files["root"], "background")
    folders_to_move = os.listdir(folder_path)

    # Creating folder and moving original files
    create_folders(background_path)
    move_all_files_and_folders(folder_path, background_path, folders_to_move)

    # Saving corrected npy files
    save_npy_files(corrected_npy_files, "no_background")


def preprocessing_pipeline(original_key_points_folder: str):
    key_points_preprocessor = KeyPointsPreProcessor()

    for folder in os.listdir(original_key_points_folder):
        # Loading .npy files
        folder_path = os.path.join(original_key_points_folder, folder)
        npy_files = load_npy_files_per_folder(folder_path)

        # Defining key-points value and adjusting npy files type
        key_points_value = get_key_points_value(KEY_POINTS_VALUES, folder)
        astype_npy_files = key_points_preprocessor.check_array_and_adjust_content_type(npy_files)

        # Checking arrays dimensions and removing background keypoint from arrays, if exists
        if astype_npy_files:
            corrected_npy_files = (key_points_preprocessor.
                                   check_array_dimensions_and_remove_background_key_point(astype_npy_files,
                                                                                          key_points_value))
        else:
            corrected_npy_files = (key_points_preprocessor.
                                   check_array_dimensions_and_remove_background_key_point(npy_files, key_points_value))

        if astype_npy_files:
            if corrected_npy_files:
                save_corrected_data(corrected_npy_files, folder_path)
            else:
                # Saving adjusted type npy files
                save_npy_files(astype_npy_files)
        elif corrected_npy_files:
            save_corrected_data(corrected_npy_files, folder_path)
