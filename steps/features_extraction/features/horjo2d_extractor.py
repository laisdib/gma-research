import numpy as np


class HORJO2DExtractor:
    def __init__(self, key_points_num: int, n_bins: int = 16):
        self.joint_pairs = self._generate_joint_pairs(key_points_num)
        self.n_bins = n_bins

    @staticmethod
    def _generate_joint_pairs(key_points_num):
        return [(i, j) for i in range(key_points_num) for j in range(i + 1, key_points_num)]

    @staticmethod
    def _calculate_orientation(p1, p2):
        return p1 - p2

    @staticmethod
    def _calculate_angle(v):
        return np.arctan2(v[1], v[0])

    def _displacement_quantization(self, features):
        # Quantization of angles in histograms with non-uniform bins
        histograms = []

        # Transpose to iterate over joints
        for joint_angles in features.T:
            bins = np.linspace(0, 360, self.n_bins + 1)

            histogram, _ = np.histogram(joint_angles, bins=bins)
            histograms.append(histogram)

        # Transpose to get the original form
        return np.array(histograms).T

    def extract_HORJO2D(self, key_points_frames: np.ndarray):
        """
        Extract Relative Joint Orientation feature from key-points.

        :param key_points_frames: np.ndarray
        :return: np.ndarray
        """

        horjo2d_features = []

        for frame in key_points_frames:
            angles = []

            for j1, j2 in self.joint_pairs:
                if j1 == j2:
                    continue

                orientation = self._calculate_orientation(frame[j1], frame[j2])
                angle = self._calculate_angle(orientation)
                angles.append(angle)

            horjo2d_features.append(angles)

        horjo2d_features = np.array(horjo2d_features)
        return self._displacement_quantization(horjo2d_features)
