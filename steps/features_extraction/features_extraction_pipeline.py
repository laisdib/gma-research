import os

from steps.features_extraction.features_extractor import FeaturesExtractor
from utils.utils import get_key_points_value, define_key_points_type, define_key_points_value
from utils.load_data import load_npy_files_per_folder
from utils.save_data import save_npy_files
from consts.key_points_info import KEY_POINTS_VALUES


def extract_all_features(id_: int, npy_files_info: dict, extracted_features_info: dict, folder_: str):
    """
    Extracting all features from key-points.

    :param id_: int
    :param npy_files_info: dict
    :param extracted_features_info: dict
    :param folder_: str
    """

    general_info = npy_files_info[id_]["content"]
    extracted_features_info.update(
        {
            id_: {
                "path": npy_files_info[id_]["path"],
                "content": []
            }
        }
    )

    # Defining key_points value and type
    key_points_value = get_key_points_value(KEY_POINTS_VALUES, folder_)
    key_points_type = define_key_points_type(KEY_POINTS_VALUES, folder_)
    key_points_value_ = define_key_points_value(key_points_value, npy_files_info[id_]["path"])

    features_extractor = FeaturesExtractor(key_points_type, key_points_value_)

    # Extracting all features
    for info in general_info:
        key_points = info["content"]
        features = features_extractor.extract_features(key_points)

        extracted_features_info[id_]["content"].append(
            {
                "file_name": info["file_name"],
                "features": features
            }
        )


def features_extraction_pipeline(normalized_key_points_folder):
    for folder in os.listdir(normalized_key_points_folder):
        # Loading .npy files
        folder_path = os.path.join(normalized_key_points_folder, folder)
        normalized_npy_files = load_npy_files_per_folder(folder_path)

        # Defining path to save features
        root_path = normalized_npy_files["root"].replace("key-points", "features")
        root_path = root_path.replace("normalized", "non_standardized")
        extracted_features = {"root": root_path}

        # Extracting and saving features
        for i in list(normalized_npy_files.keys())[1:]:
            extract_all_features(i, normalized_npy_files, extracted_features, folder)

        save_npy_files(extracted_features, is_features=True)
