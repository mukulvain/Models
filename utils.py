import numpy as np

def load_data(data_config):
    """
    Reads data from a csv file and returns X and Y separately.
    """
    data_file = data_config["file_path"]
    feature_cols = [i for i in range(int(data_config["feature_cols"]))]
    label_col = int(data_config["label_col"])

    data = np.genfromtxt(data_file, delimiter = ",")

    return data[:,feature_cols], data[:,label_col]