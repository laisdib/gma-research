import os

from df_generator import DataFrameGenerator
from utils.utils import define_dataset_path


def generate_std_dataframe(standardized_folder_path_: str, dataset_: str | None = None):
    """
    Generate dataframe with standardized features info.

    :param standardized_folder_path_: str
    :param dataset_: str | None
    """

    standardized_dataset_path_ = define_dataset_path(standardized_folder_path_, dataset_)
    standardized_dfs_folder_ = standardized_dataset_path_.replace("standardized", "dataframes")

    data_loader_ = DataFrameGenerator(standardized_dataset_path_, unit_features)
    data_loader_.generate_features_dataframe(standardized_dfs_folder_)


def generate_fused_dataframe(fused_folder_path_: str, dataset_: str | None = None):
    """
    Generate dataframe with fused features info.

    :param fused_folder_path_: str
    :param dataset_: str | None
    """

    fused_dataset_path_ = define_dataset_path(fused_folder_path_, dataset_)
    fused_dfs_folder_ = fused_dataset_path_.replace("fused", "dataframes")

    data_loader_ = DataFrameGenerator(fused_dataset_path_, fused_features)
    data_loader_.generate_features_dataframe(fused_dfs_folder_)


root_path = "../data/features"
standardized_features_folder = f"{root_path}/standardized"
fused_features_folder = f"{root_path}/fused"

unit_features = ["FFT-JD", "FFT-JO", "HOAD2D", "HOJD2D", "HOJO2D", "HORJAD2D", "HORJO2D"]
fused_features = ["pose_based", "velocity_based", "all_features"]

for folder in os.listdir(standardized_features_folder):
    standardized_folder_path = os.path.join(standardized_features_folder, folder)
    fused_folder_path = os.path.join(fused_features_folder, folder)

    datasets = os.listdir(standardized_folder_path)

    # Generating and saving dataframes with standardized and fused features info
    if "treino" in datasets:
        generate_std_dataframe(standardized_folder_path)
        generate_fused_dataframe(fused_folder_path)
    else:
        for dataset in datasets:
            generate_std_dataframe(standardized_folder_path, dataset)
            generate_fused_dataframe(fused_folder_path, dataset)
