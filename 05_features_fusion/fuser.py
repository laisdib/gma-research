import os
import numpy as np


class FeaturesFuser:
    def __init__(self, data: dict):
        self.data = data
        self.pose_based = ["HOJO2D", "HOAD2D", "HORJO2D", "FFT-JO"]
        self.velocity_based = ["HOJD2D", "HORJAD2D", "FFT-JD"]

    def _define_features_list(self, features):
        # Defines features list
        pose_based_features = []
        velocity_based_features = []

        for key, value in features.items():
            key = os.path.splitext(key)[0]

            if key in self.pose_based:
                pose_based_features.append(value)
            elif key in self.velocity_based:
                velocity_based_features.append(value)

        return pose_based_features, velocity_based_features

    @staticmethod
    def _apply_zero_padding(shapes, features):
        # Applies zero lines at the end of the matrix, if necessary
        amount_samples = [shape[0] for shape in shapes]
        max_samples = max(amount_samples)
        padded_features = {}

        for feature_name, feature in features.items():
            padding_needed = max_samples - feature.shape[0]

            if padding_needed > 0:
                padded_feature = np.pad(feature, ((0, padding_needed), (0, 0)), mode='constant', constant_values=0)
            else:
                padded_feature = feature

            padded_features.update({feature_name: padded_feature})

        return padded_features

    def fuse_features(self, dataset: str | None = None):
        """
        Fuses features according pose-based and velocity-based features.

        :param dataset: str | None
        :return: dict
        """

        root_path = self.data["root"].replace("standardized", "fused")
        fused_features = {"root": root_path}

        for i in list(self.data.keys())[1:]:
            dataset_name = self.data[i]["path"]
            dataset_name = os.path.dirname(os.path.dirname(dataset_name))

            if (dataset is not None) and (dataset != dataset_name):
                continue

            general_info = self.data[i]["content"]
            fused_features.update(
                {
                    i: {
                        "path": self.data[i]["path"],
                        "content": []
                    }
                }
            )

            for info in general_info:
                features = info["features"]
                shapes = {feature.shape for feature in features.values()}

                if len(shapes) > 1:
                    features = self._apply_zero_padding(shapes, features)

                pose_based_features_list, velocity_based_features_list = self._define_features_list(features)

                pose_based_features = np.concatenate(pose_based_features_list, axis=1)
                velocity_based_features = np.concatenate(velocity_based_features_list, axis=1)

                fused_features[i]["content"].append(
                    {
                        "file_name": info["file_name"],
                        "features": {
                            "pose_based": pose_based_features,
                            "velocity_based": velocity_based_features,
                            "all_features": np.concatenate([pose_based_features, velocity_based_features], axis=1)
                        }
                    }
                )

        return fused_features
