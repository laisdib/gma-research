import numpy as np


class HOAD2DExtractor:
    def __init__(self, parent_joints_pair: list, n_bins: int = 16, offset: int = 10):
        self.parent_joints_pair = parent_joints_pair
        self.n_bins = n_bins
        self.offset = offset

    @staticmethod
    def _calculate_orientation(p1, p2):
        return p1 - p2

    @staticmethod
    def _calculate_angle(v1, v2):
        cosine_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        # Ensure the value is within the allowed range [-1, 1]
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        return np.arccos(cosine_similarity)

    def _displacement_quantization(self, features):
        # Quantization of angles in histograms with non-uniform bins
        histograms = []

        # Transpose to iterate over joints
        for joint_angles in features.T:
            max_angle = np.pi  # Theoretical maximum angle
            bins = [max_angle / (2 ** bin_) for bin_ in range(self.n_bins)]
            bins.sort()

            histogram, _ = np.histogram(joint_angles, bins=bins)
            histograms.append(histogram)

        # Transpose to get the original form
        return np.array(histograms).T

    def extract_HOAD2D(self, key_points_frames: np.ndarray):
        """
        Extract Angular Displacement feature from key-points.

        :param key_points_frames: np.ndarray
        :return: np.ndarray
        """

        hoad2d_features = []

        for i in range(len(key_points_frames) - self.offset):
            frame = key_points_frames[i]
            frame_offset = key_points_frames[i + self.offset]

            angles = []

            for parent, child in self.parent_joints_pair:
                orientation_current = self._calculate_orientation(frame[child], frame[parent])
                orientation_offset = self._calculate_orientation(frame_offset[child], frame_offset[parent])

                angle = self._calculate_angle(orientation_current, orientation_offset)
                angles.append(angle)

            hoad2d_features.append(angles)

        hoad2d_features = np.array(hoad2d_features)
        return self._displacement_quantization(hoad2d_features)
