import numpy as np

from consts import mediapipe_info, openpose_info
from utils.utils import define_key_points_value


class KeyPointsNormalizer:
    def __init__(self, key_points_type: str):
        self.key_points_type = key_points_type

        if self.key_points_type == "OpenPose":
            self.body_parts = openpose_info.OPENPOSE
        elif self.key_points_type == "MediaPipe":
            self.body_parts = mediapipe_info.MEDIAPIPE

    def _calculate_spine_vector_and_norm(self, frame: np.ndarray, root_key_point: int | float):
        """
        Calculate spine vector and norm.

        :param frame: np.array
        :param root_key_point: int | float
        :return: tuple[np.ndarray, float | np.ndarray]
        """

        if self.key_points_type == "OpenPose":
            neck_position = self.body_parts["Neck"]
        else:
            neck_position = self.body_parts["nose"]

        neck_key_point = frame[neck_position]
        spine_vector = neck_key_point - root_key_point
        norm_spine_vector = np.linalg.norm(spine_vector)

        return spine_vector, norm_spine_vector

    def _calculate_angle_direction(self, frame: np.ndarray, root_key_point: int | float):
        """
        Calculates the angle to align the column (line between the "Neck" and "RHip" joints) with the y-axis.

        :param frame: np.array
        :param root_key_point: int | float
        :return: float | None
        """

        spine_vector, norm_spine_vector = self._calculate_spine_vector_and_norm(frame, root_key_point)

        if norm_spine_vector == 0:
            return

        y_axis = np.array([0, 1])

        cos_theta = np.dot(spine_vector, y_axis) / norm_spine_vector
        cos_theta = np.clip(cos_theta, -1, 1)  # To avoid numerical errors outside the domain
        angle = np.arccos(cos_theta)

        # Determine the angle direction
        dir_sign = np.sign(np.dot(spine_vector, y_axis))
        angle *= dir_sign

        return angle

    def normalize_key_points(self, data: dict, key_points_num: dict | int):
        """
        Normalize key-points.

        :param data: dict
        :param key_points_num: dict | int
        :return: dict
        """

        root_path = data["root"].replace("original", "normalized")
        normalized_key_points = {"root": root_path}

        for i in list(data.keys())[1:]:
            general_info = data[i]["content"]
            normalized_key_points.update(
                {
                    i: {
                        "path": data[i]["path"],
                        "content": []
                    }
                }
            )

            for info in general_info:
                key_points_num_ = define_key_points_value(key_points_num, data[i]["path"])

                if info["shape"][2] < 2 or info["shape"][2] > 3:
                    raise ValueError(f"Unexpected key-point format. Expected (900, {key_points_num_}, 2) or "
                                     f"(900, {key_points_num_}, 3).")

                key_points = info["content"].copy()

                for frame in key_points:
                    if self.key_points_type == "OpenPose":
                        root_position = self.body_parts["RHip"]
                    else:
                        root_position = self.body_parts["right hip"]

                    root_key_point = frame[root_position]

                    # Translates the root point to the origin (0, 0)
                    frame -= root_key_point

                    # Calculates the column alignment angle
                    angle = self._calculate_angle_direction(frame, root_key_point)

                    if not angle:
                        continue

                    # Rotation matrix
                    rotation_matrix = np.array([
                        [np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]
                    ])

                    # Applies rotation to all joints
                    for j in range(len(frame)):
                        frame[j] = np.dot(rotation_matrix, frame[j])

                normalized_key_points[i]["content"].append(
                    {
                        "file_name": info["file_name"],
                        "content": key_points,
                        "shape": key_points.shape
                    }
                )

        return normalized_key_points
