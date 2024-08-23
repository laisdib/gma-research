import numpy as np

class HORJAD2DExtractor:
    def __init__(self, key_points_num: int, n_bins: int = 8, offset: int = 10):
        self.joint_pairs = self._generate_joint_pairs(key_points_num)
        self.n_bins = n_bins
        self.offset = offset

    @staticmethod
    def _generate_joint_pairs(key_points_num):
        return [(i, j) for i in range(key_points_num) for j in range(i + 1, key_points_num)]

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

        for joint_angles in features.T:
            max_angle = np.pi  # Theoretical maximum angle
            bins = [max_angle / (2 ** bin_) for bin_ in range(self.n_bins)]
            bins.sort()

            histogram, _ = np.histogram(joint_angles, bins=bins)
            histograms.append(histogram)

        return np.array(histograms).T

    def extract_HORJAD2D(self, key_points_frames: np.ndarray):
        """
        Extract Relative Joint Angular Displacement feature from key-points.

        :param key_points_frames: np.ndarray
        :return: np.ndarray
        """

        horjad2d_features = []

        for i in range(len(key_points_frames) - self.offset):
            frame = key_points_frames[i]
            frame_offset = key_points_frames[i + self.offset]

            angles = []

            for j1, j2 in self.joint_pairs:
                orientation_current = self._calculate_orientation(frame[j1], frame[j2])
                orientation_offset = self._calculate_orientation(frame_offset[j1], frame_offset[j2])

                angle = self._calculate_angle(orientation_current, orientation_offset)
                angles.append(angle)

            horjad2d_features.append(angles)

        horjad2d_features = np.array(horjad2d_features)
        return self._displacement_quantization(horjad2d_features)
