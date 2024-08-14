class KeyPointsPreProcessor:
    def __init__(self):
        self.accepted_contents = ["float32", "float64", "int32", "int64"]

    @staticmethod
    def _remove_background_key_point(content: dict, key_points_num: int):
        """
        Remove background key-point.

        :param content: dict
        :param key_points_num: int
        :return: dict
        """

        nparray = content["content"]
        nparray = nparray[:, :key_points_num, :]

        new_content = {
            "file_name": content["file_name"],
            "content": nparray,
            "shape": nparray.shape
        }

        return new_content

    def _adjust_key_points_type(self, content: dict):
        """
        Adjust key-points type.

        :param content: dict
        :return: dict
        """

        nparray = content["content"]

        if nparray.dtype not in self.accepted_contents:
            nparray = nparray.astype(float)

        new_content = {
            "file_name": content["file_name"],
            "content": nparray,
            "shape": nparray.shape
        }

        return new_content

    def check_array_dimensions_and_remove_background_key_point(self, data: dict, key_points_num: int | None):
        """
        Check array dimensions and remove background key-point, if exists.

        :param data: dict
        :param key_points_num: int | None
        :return: dict | None
        """

        changed_data = {"root": data["root"]}

        for i in list(data.keys())[1:]:
            general_info = data[i]["content"]
            changed_data.update(
                {
                    i: {
                        "path": data[i]["path"],
                        "content": []
                    }
                }
            )

            if key_points_num is None:
                return

            for info in general_info:
                if info["shape"][1] <= key_points_num:
                    continue

                # Removing background keypoint from arrays, if exists
                new_content = self._remove_background_key_point(info, key_points_num)
                changed_data[i]["content"].append(new_content)

        for i in list(changed_data.keys())[1:]:
            general_info = changed_data[i]["content"]

            for j in range(len(general_info)):
                if general_info[j]["shape"] == data[i]["content"][j]["shape"]:
                    continue

                return changed_data

        return

    def check_array_and_adjust_content_type(self, data: dict):
        """
        Check array and adjust key-points type.

        :param data: dict
        :return: dict | None
        """

        changed_data = {"root": data["root"]}

        for i in list(data.keys())[1:]:
            general_info = data[i]["content"]
            changed_data.update(
                {
                    i: {
                        "path": data[i]["path"],
                        "content": []
                    }
                }
            )

            for info in general_info:
                new_content = self._adjust_key_points_type(info)
                changed_data[i]["content"].append(new_content)

        for i in list(changed_data.keys())[1:]:
            general_info = changed_data[i]["content"]

            for j in range(len(general_info)):
                if general_info[j]["content"].dtype == data[i]["content"][j]["content"].dtype:
                    continue

                return changed_data

        return
