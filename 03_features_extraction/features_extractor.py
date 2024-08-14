import numpy as np

from consts import mediapipe_info, openpose_info
from features.hoad2d_extractor import HOAD2DExtractor
from features.horjo2d_extractor import HORJO2DExtractor
from features.horjad2d_extractor import HORJAD2DExtractor
from features.fft_jd_extractor import FFTJDExtractor
from features.fft_jo_extractor import FFTJOExtractor
from features.hojo2d_extractor import HOJO2DExtractor
from features.hojd2d_extractor import HOJD2DExtractor


class FeaturesExtractor:
    def __init__(self, key_points_type: str, key_points_num: int):
        self.features_names = ["HOAD2D", "HORJO2D", "HORJAD2D", "FFT-JD", "FFT-JO", "HOJO2D", "HOJD2D"]
        self.key_points_type = key_points_type
        self.key_points_num = key_points_num

        if self.key_points_type == "OpenPose":
            self.parent_joints_pair = openpose_info.OPENPOSE_PARENT_JOINT_PAIR
        else:
            self.parent_joints_pair = mediapipe_info.MEDIAPIPE_PARENT_JOINT_PAIR

    def _generate_features_info(self, features: list[np.ndarray]):
        """
        Generates features info dict.

        :param features: list[np.ndarray]
        :return: dict
        """

        features_info = {}

        for i, name in enumerate(self.features_names):
            features_info.update({name: features[i]})

        return features_info

    def extract_features(self, data: np.ndarray):
        """
        Extract features from key-points np.ndarray.

        :param data: np.ndarray
        :return: dict
        """

        hoad2d_extractor = HOAD2DExtractor(self.parent_joints_pair)
        hoad2d = hoad2d_extractor.extract_HOAD2D(data)

        horjo2d_extractor = HORJO2DExtractor(self.key_points_num)
        horjo2d = horjo2d_extractor.extract_HORJO2D(data)

        horjad2d_extractor = HORJAD2DExtractor(self.key_points_num)
        horjad2d = horjad2d_extractor.extract_HORJAD2D(data)

        fft_jd_extractor = FFTJDExtractor()
        fft_jd = fft_jd_extractor.extract_FFT_JD(data)

        fft_jo_extractor = FFTJOExtractor()
        fft_jo = fft_jo_extractor.extract_FFT_JO(data)

        hojo2d_extractor = HOJO2DExtractor(self.key_points_num)
        hojo2d = hojo2d_extractor.extract_HOJO2D(data)

        hojd2d_extractor = HOJD2DExtractor(self.key_points_num)
        hojd2d = hojd2d_extractor.extract_HOJD2D(data)

        features = [hoad2d, horjo2d, horjad2d, fft_jd, fft_jo, hojo2d, hojd2d]
        features_info = self._generate_features_info(features)

        return features_info
