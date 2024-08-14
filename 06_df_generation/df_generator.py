import os
import pandas as pd

from utils.save_data import create_folders


class DataFrameGenerator:
    def __init__(self, dataset_root_path, features_list):
        self.dataset_root_path = dataset_root_path
        self.features_list = features_list

    @staticmethod
    def _save_dataframe(features_info: dict, dataframe_path: str, feature_name: str):
        # Saves dataframe with features info
        df = pd.DataFrame(features_info)
        create_folders(dataframe_path)
        df.to_csv(os.path.join(dataframe_path, f"{feature_name}.csv"), index=False)

    def generate_features_dataframe(self, dataframe_path: str):
        """
        Generate features dataframe.

        :param dataframe_path: str
        """

        for feature in self.features_list:
            for dataset_part in os.listdir(self.dataset_root_path):
                features_info = {
                    "video_name": [],
                    "feature_path": [],
                    "label": []
                }

                dataset_part_path = os.path.join(self.dataset_root_path, dataset_part)

                for label in os.listdir(dataset_part_path):
                    label_path = os.path.join(dataset_part_path, label)

                    for video in os.listdir(label_path):
                        feature_path = os.path.join(label_path, video, f"{feature}.npy")

                        features_info["video_name"].append(video)
                        features_info["feature_path"].append(feature_path)
                        features_info["label"].append(label)

                save_df_path = os.path.join(dataframe_path, dataset_part)
                self._save_dataframe(features_info, save_df_path, feature)
