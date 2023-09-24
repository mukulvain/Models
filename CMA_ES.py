import configparser
import pickle
import sys

import cma
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils import load_data


class CMA_ES:
    def __init__(self, config_file):
        # Parse the config file.
        config = configparser.ConfigParser()
        config.read(config_file)

        # CMA_ES parameters.
        cma_es_config = config["CMA_ES"]
        self.iterations = int(cma_es_config.get("iterations", 100))
        self.population_size = int(cma_es_config.get("population_size", 20))
        self.initial_sigma = float(cma_es_config.get("initial_sigma", 0.1))
        self.options = cma.CMAOptions()
        self.options.set("popsize", self.population_size)
        self.options.set("maxiter", self.iterations)

        # SVR parameters.
        svr_config = config["SVR"]
        self.kernel = svr_config.get("kernel", "linear")
        self.epsilon = float(svr_config.get("epsilon", 0.01))
        self.C_min = float(svr_config.get("C_min", 0.01))
        self.C_max = float(svr_config.get("C_max", 100))
        self.gamma_min = float(svr_config.get("gamma_min", 0.01))
        self.gamma_max = float(svr_config.get("gamma_max", 10))
        self.options.set("BoundaryHandler", cma.BoundPenalty)
        self.options.set(
            "bounds", [(self.C_min, self.gamma_min), (self.C_max, self.gamma_max)]
        )

        # Data file
        data_config = config["Data"]

        # Load the data
        X, Y = load_data(data_config)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        X_scaler = StandardScaler()
        self.X_train = X_scaler.fit_transform(self.X_train)
        self.X_test = X_scaler.transform(self.X_test)

    # Define the SVR objective function to be minimized by CMA_ES
    def svr(self, params):
        C, gamma = params
        svr_model = SVR(C=C, epsilon=self.epsilon, gamma=gamma, kernel=self.kernel)
        svr_model.fit(self.X_train, self.Y_train)
        Y_pred = svr_model.predict(self.X_test)

        # You can use any regression metric like Mean Squared Error (MSE) here
        MSE = np.mean((self.Y_test - Y_pred) ** 2)
        return MSE

    def run_optimizer(self):
        self.es = cma.CMAEvolutionStrategy([1, 1], self.initial_sigma, self.options)
        self.es.optimize(self.svr)

        print("Best Parameters (C, gamma):", self.es.best.x)
        self.best_params = self.es.best.x

    def train_svr(self):
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

        save_path = config["Model"]["save_path"]
        with open(save_path, "wb") as save_file:
            pickle.dump(best_svr_model, save_file)


if __name__ == "__main__":
    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)

    print("Initialising CMA_ES...")
    optimizer = CMA_ES(config_file)

    print("Optimizing SVR using CMA_ES...\n")
    optimizer.run_optimizer()

    print("Training SVR using Best Params...")
    optimizer.train_svr()
