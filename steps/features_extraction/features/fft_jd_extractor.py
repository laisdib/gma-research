import numpy as np


class FFTJDExtractor:
    def __init__(self, n_bins: int = 16):
        self.n_bins = n_bins

    @staticmethod
    def _calculate_joint_displacement(key_points_frames: np.ndarray):
        # Calculates the joints displacement between consecutive frames
        return np.diff(key_points_frames, axis=0)

    @staticmethod
    def _apply_fft(displacement: np.ndarray):
        # Applies FFT to joints displacement signal
        return np.abs(np.fft.fft(displacement, axis=0))

    def _frequency_quantization(self, frequency_components: np.ndarray):
        # Quantization of angles in histograms with non-uniform bins
        histograms = []
        F = frequency_components.shape[0]

        for component in frequency_components.T:
            bins = [F * (i ** 2) / self.n_bins ** 2 for i in range(1, self.n_bins + 1)]

            histogram, _ = np.histogram(component, bins=bins)
            histograms.append(histogram)

        return np.array(histograms).T

    def extract_FFT_JD(self, key_points_frames: np.ndarray):
        """
        Extract Fast Fourier Transform of Joint Displacement feature from key-points.

        :param key_points_frames: np.ndarray
        :return: np.ndarray
        """

        joint_displacements = self._calculate_joint_displacement(key_points_frames)
        fft_components = self._apply_fft(joint_displacements)
        fft_jd_features = self._frequency_quantization(fft_components)

        return fft_jd_features
