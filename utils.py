import numpy as np


def load_data(data_config):
    """
    Reads data from a csv file and returns X and Y separately.
    """
    data_file = data_config["file_path"]
    feature_cols = [i for i in range(int(data_config["feature_cols"]))]
    label_col = int(data_config["label_col"])

    data = np.genfromtxt(data_file, delimiter=",")

    return data[:, feature_cols], data[:, label_col]


def locate(coordinates, angles, distance):
    """
    Takes the following as parameters:
    - coordinates of the sensor
    - angles observed by the sensor (in degrees)
    - distance measured using RSSI values

    Returns:
    - coordinates of the object

    Refer:
    https://www.geogebra.org/m/yj2sVvNJ
    """
    assert len(coordinates) == 3
    assert len(angles) == 3

    position = []
    for i in range(len(coordinates)):
        position.append(coordinates[i] + distance * np.cos(np.deg2rad(angles[i])))

    return position
