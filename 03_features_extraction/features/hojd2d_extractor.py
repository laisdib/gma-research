import numpy as np

class HOJD2DExtractor:
    def __init__(self, key_points_num: int, n_bins: int = 16, step: int = 5):
        self.joint_pairs = self._generate_joint_pairs(key_points_num)
        self.n_bins = n_bins
        self.step = step

    @staticmethod
    def _generate_joint_pairs(key_points_num):
        return [(i, j) for i in range(key_points_num) for j in range(i + 1, key_points_num)]

    def _calculate_joint_displacement(self, key_points_frames: np.ndarray):
        displacements = []

        for i in range(0, len(key_points_frames) - self.step, self.step):
            displacement = (key_points_frames[i + self.step, self.joint_pairs, :]
                            - key_points_frames[i, self.joint_pairs, :])
            displacements.append(displacement)

        return np.array(displacements)

    def _quantize_displacements(self, displacements):
        # Quantization of displacements in histograms with uniform bins
        bins = np.linspace(0, np.max(displacements), self.n_bins + 1)
        histograms = []

        # Transpose to iterate over joints
        for joint_displacement in displacements.T:
            histogram, _ = np.histogram(joint_displacement, bins=bins)
            histograms.append(histogram)

        # Transpose to get the original form
        return np.array(histograms).T

    def extract_HOJD2D(self, key_points_frames: np.ndarray):
        """
        Extract Histogram of Joint Displacement feature from key-points.

        :param key_points_frames: np.ndarray
        :return: np.ndarray
        """

        joint_displacements = self._calculate_joint_displacement(key_points_frames)
        hojd2d_features = self._quantize_displacements(joint_displacements)

        return hojd2d_features
