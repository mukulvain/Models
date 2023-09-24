import configparser
import pickle
import sys
import numpy as np
import pyswarms as ps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils import load_data


class PSO:
    def __init__(self, config_file):
        # Parse the config file.
        config = configparser.ConfigParser()
        config.read(config_file)

        # PSO parameters.
        pso_config = config["PSO"]
        self.iterations = int(pso_config.get("iterations", 100))
        self.swarm_size = int(pso_config.get("swarmsize", 10))
        self.inertia_wt = float(pso_config.get("inertia_weight", 1))
        self.c1 = float(pso_config.get("c1", 2))
        self.c2 = float(pso_config.get("c2", 2))
        self.options = {"c1": self.c1, "c2": self.c2, "w": self.inertia_wt}

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
        X, Y = load_data(data_config)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        X_scaler = StandardScaler()
        self.X_train = X_scaler.fit_transform(self.X_train)
        self.X_test = X_scaler.transform(self.X_test)

    # Define the SVR objective function to be minimized by PSO
    def svr(self, params):
        errors = []

        for i in range(len(params)):
            C, gamma = params[i]
            svr_model = SVR(C=C, epsilon=self.epsilon, gamma=gamma, kernel=self.kernel)
            svr_model.fit(self.X_train, self.Y_train)
            Y_pred = svr_model.predict(self.X_test)

            # You can use any regression metric like Mean Squared Error (MSE) here
            MSE = np.mean((self.Y_test - Y_pred) ** 2)
            errors.append(MSE)

        return np.array(errors)

    def run_optimizer(self):
        # Define the parameter bounds for PSO
        lower_bound = [self.C_min, self.gamma_min]  # Lower bounds for C, gamma
        upper_bound = [self.C_max, self.gamma_max]  # Upper bounds for C, gamma
        bounds = (lower_bound, upper_bound)

        # Use PSO to optimize SVR parameters
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.swarm_size,
            dimensions=2,
            options=self.options,
            bounds=bounds,
        )

        self.best_cost, self.best_params = optimizer.optimize(
            self.svr, iters=self.iterations, n_processes=self.swarm_size
        )

        # Print the best parameters found by PSO
        print("Best Parameters (C, gamma):", self.best_params)

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

    print("Initialising PSO...")
    optimizer = PSO(config_file)

    print("Optimizing SVR using PSO...\n")
    optimizer.run_optimizer()

    print("Training SVR using Best Params...")
    optimizer.train_svr()
