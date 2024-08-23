import os
import numpy as np


class FeaturesStandardizer:
    def __init__(self, data: dict, dataset: str | None = None):
        self.data = data
        self.all_features = self._load_all_features(dataset)
        self.means, self.stds = self._calculate_mean_and_std()

    def _load_all_features(self, dataset: str | None):
        # Loads all features
        all_features = {}

        for i in list(self.data.keys())[1:]:
            dataset_name = self.data[i]["path"]
            dataset_name = os.path.dirname(os.path.dirname(dataset_name))

            if (dataset is not None) and (dataset != dataset_name):
                continue

            general_info = self.data[i]["content"]

            for info in general_info:
                for feature_name, feature in info["features"].items():
                    if feature_name not in list(all_features.keys()):
                        all_features.update({feature_name: []})

                    all_features[feature_name].append(feature)

        return all_features

    def _calculate_mean_and_std(self):
        # Calculates the features mean and standard deviation
        means = {}
        stds = {}

        for feature_name, feature in self.all_features.items():
            mean = np.mean(feature, axis=0)
            std = np.std(feature, axis=0)

            means.update({feature_name: mean})
            stds.update({feature_name: std})

        return means, stds

    def _z_score_normalization(self, feature_name, features):
        # Applies Z-score normalization on features
        mean = self.means[feature_name]
        std = self.stds[feature_name]

        # Avoids division by zero
        std_adj = np.where(std == 0, 1, std)

        normalized_features = (features - mean) / std_adj

        # Replaces NaN by zero after normalization
        normalized_features = np.where(std == 0, 0, normalized_features)

        return normalized_features

    def standardize_features(self, dataset: str | None):
        """
        Standardizes features according Z-score normalization.

        :param dataset: str | None
        :return: dict
        """

        root_path = self.data["root"].replace("non_standardized", "standardized")
        standardized_features = {"root": root_path}

        for i in list(self.data.keys())[1:]:
            dataset_name = self.data[i]["path"]
            dataset_name = os.path.dirname(os.path.dirname(dataset_name))

            if (dataset is not None) and (dataset != dataset_name):
                continue

            general_info = self.data[i]["content"]
            standardized_features.update(
                {
                    i: {
                        "path": self.data[i]["path"],
                        "content": []
                    }
                }
            )

            for info in general_info:
                features = info["features"]
                final_features = {}

                for key, value in features.items():
                    feature = self._z_score_normalization(key, value)
                    final_features.update({key: feature})

                standardized_features[i]["content"].append(
                    {
                        "file_name": info["file_name"],
                        "features": final_features
                    }
                )

        return standardized_features
