import numpy as np


class HOJO2DExtractor:
    def __init__(self, key_points_num: int, n_bins: int = 16):
        self.joint_pairs = self._generate_joint_pairs(key_points_num)
        self.n_bins = n_bins

    @staticmethod
    def _generate_joint_pairs(key_points_num):
        return [(i, j) for i in range(key_points_num) for j in range(i + 1, key_points_num)]

    @staticmethod
    def _calculate_orientation(p1, p2):
        return p2 - p1

    @staticmethod
    def _calculate_angle(orientation):
        angle = np.degrees(np.arctan2(orientation[1], orientation[0]))
        return angle % 360  # Normalizes angle

    def _quantize_orientations(self, angles):
        # Quantization of orientations in histograms with uniform bins
        bins = np.linspace(0, 360, self.n_bins + 1)
        histograms = []

        # Transpose to iterate over joints
        for joint_angles in angles.T:
            histogram, _ = np.histogram(joint_angles, bins=bins)
            histograms.append(histogram)

        # Transpose to get the original form
        return np.array(histograms).T

    def extract_HOJO2D(self, key_points_frames: np.ndarray):
        """
        Extract Histogram of Joint Orientation feature from key-points.

        :param key_points_frames: np.ndarray
        :return: np.ndarray
        """

        hojo2d_features = []

        for frame in key_points_frames:
            angles = []

            for (i, j) in self.joint_pairs:
                p1, p2 = frame[i], frame[j]
                orientation = self._calculate_orientation(p1, p2)

                angle = self._calculate_angle(orientation)
                angles.append(angle)

            angles = np.array(angles)
            hojo2d_features.append(angles)

        hojo2d_features = np.array(hojo2d_features)
        return self._quantize_orientations(hojo2d_features)
