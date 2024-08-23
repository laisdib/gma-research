import os

from steps.df_generation.df_generator import DataFrameGenerator
from utils.utils import define_dataset_path


def generate_std_dataframe(standardized_folder_path_: str, unit_features: list, dataset_: str | None = None):
    """
    Generate dataframe with standardized features info.

    :param standardized_folder_path_: str
    :param unit_features: list
    :param dataset_: str | None
    """

    standardized_dataset_path_ = define_dataset_path(standardized_folder_path_, dataset_)
    standardized_dfs_folder_ = standardized_dataset_path_.replace("standardized", "dataframes")

    data_loader_ = DataFrameGenerator(standardized_dataset_path_, unit_features)
    data_loader_.generate_features_dataframe(standardized_dfs_folder_)


def generate_fused_dataframe(fused_folder_path_: str, fused_features: list, dataset_: str | None = None):
    """
    Generate dataframe with fused features info.

    :param fused_folder_path_: str
    :param fused_features: list
    :param dataset_: str | None
    """

    fused_dataset_path_ = define_dataset_path(fused_folder_path_, dataset_)
    fused_dfs_folder_ = fused_dataset_path_.replace("fused", "dataframes")

    data_loader_ = DataFrameGenerator(fused_dataset_path_, fused_features)
    data_loader_.generate_features_dataframe(fused_dfs_folder_)


def df_generation_pipeline(standardized_features_folder: str, fused_features_folder: str, unit_features: list,
                           fused_features: list):
    for folder in os.listdir(standardized_features_folder):
        standardized_folder_path = os.path.join(standardized_features_folder, folder)
        fused_folder_path = os.path.join(fused_features_folder, folder)

        datasets = os.listdir(standardized_folder_path)

        # Generating and saving dataframes with standardized and fused features info
        if "treino" in datasets:
            generate_std_dataframe(standardized_folder_path, unit_features)
            generate_fused_dataframe(fused_folder_path, fused_features)
        else:
            for dataset in datasets:
                generate_std_dataframe(standardized_folder_path, unit_features, dataset)
                generate_fused_dataframe(fused_folder_path, fused_features, dataset)
