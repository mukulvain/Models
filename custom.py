import configparser
import pickle
import sys

import numpy as np
from mealpy.evolutionary_based import GA
from mealpy.bio_based import SMA
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils import load_data


class Custom:
    def __init__(self, config_file, model):
        # Parse the config file.
        config = configparser.ConfigParser()
        config.read(config_file)

        # Custom Model
        self.model = model

        # SVR parameters.
        svr_config = config["SVR"]
        self.kernel = svr_config.get("kernel", "linear")
        self.epsilon = float(svr_config.get("epsilon", 0.01))
        self.C_min = float(svr_config.get("C_min", 0.01))
        self.C_max = float(svr_config.get("C_max", 100))
        self.gamma_min = float(svr_config.get("gamma_min", 0.01))
        self.gamma_max = float(svr_config.get("gamma_max", 10))

        # Data file
        data_config = config["Data"]

        # Load the data
        # X, Y = load_data(data_config)
        X, Y = load_diabetes(return_X_y=True)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        X_scaler = StandardScaler()
        self.X_train = X_scaler.fit_transform(self.X_train)
        self.X_test = X_scaler.transform(self.X_test)

    # Define the SVR objective function to be minimized by Custom Algorithm
    def svr(self, params):
        C, gamma = params
        svr_model = SVR(C=C, epsilon=self.epsilon, gamma=gamma, kernel=self.kernel)
        svr_model.fit(self.X_train, self.Y_train)
        Y_pred = svr_model.predict(self.X_test)

        # You can use any regression metric like Mean Squared Error (MSE) here
        MSE = np.mean((self.Y_test - Y_pred) ** 2)
        return MSE

    def run_optimizer(self):
        problem_dict = {
            "fit_func": self.svr,
            "lb": [self.C_min, self.gamma_min],
            "ub": [self.C_max, self.gamma_max],
            "minmax": "min",
        }

        self.best_params, _ = self.model.solve(problem_dict)
        print("Best Parameters (C, gamma):", self.best_params)

    def train_svr(self, save_path):
        # Train an SVR model with the best parameters
        best_C, best_gamma = self.best_params
        best_svr_model = SVR(
            C=best_C, epsilon=self.epsilon, gamma=best_gamma, kernel=self.kernel
        )
        best_svr_model.fit(self.X_train, self.Y_train)

        # Evaluate the SVR model
        Y_pred = best_svr_model.predict(self.X_test)
        MSE = np.mean((self.Y_test - Y_pred) ** 2)

        print("Final MSE:", MSE)

        with open(save_path, "wb") as save_file:
            pickle.dump(best_svr_model, save_file)


if __name__ == "__main__":
    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)

    """
    Create your Model here, in the following way:

    model = Model_Name(parameters) 
    """

    # model = GA.BaseGA(epoch=100, pop_size=50, pc=0.85, pm=0.1)
    model = SMA.BaseSMA(epoch=100, pop_size=50, pr=0.03)

    print("Initialising Custom Algorithm...")
    optimizer = Custom(config_file, model)

    print("Optimizing SVR using Custom Algorithm...\n")
    optimizer.run_optimizer()

    print("Training SVR using Best Params...")
    optimizer.train_svr("models/Model_Name.pickle")
