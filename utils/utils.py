import os


def get_key_points_value(key_points_values: dict, folder: str):
    """
    Get key-point value according dataset.

    :param key_points_values: dict
    :param folder: str
    :return: None | int | dict
    """

    key_points_value = None

    for key, value in key_points_values.items():
        if key in folder:
            key_points_value = value

    return key_points_value


def define_key_points_type(key_points_values: dict, folder: str):
    """
    Define key-point type according dataset.

    :param key_points_values: dict
    :param folder: str
    :return: None | int
    """

    key_points_type = None

    for key in list(key_points_values.keys()):
        if key in folder:
            key_points_type = key

    return key_points_type


def define_key_points_value(key_points_value: dict | int, key_points_path: str):
    """
    Define key-points value according dataset.

    :param key_points_value: dict | int
    :param key_points_path: str
    :return: int
    """

    if type(key_points_value) is dict:
        folder = os.path.dirname(os.path.dirname(key_points_path))
        key_points_value_ = key_points_value[folder]
    else:
        key_points_value_ = key_points_value

    return key_points_value_


def define_dataset_path(folder_path: str, dataset: str | None):
    """
    Define dataset path.

    :param folder_path: str
    :param dataset: str | None
    :return: str
    """

    if dataset is None:
        dataset_path = folder_path
    else:
        dataset_path = os.path.join(folder_path, dataset)

    return dataset_path
