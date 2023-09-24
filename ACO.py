import configparser
import pickle
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils import load_data


class ACO:
    def __init__(self, config_file):
        # Parse the config file.
        config = configparser.ConfigParser()
        config.read(config_file)

        # ACO parameters.
        aco_config = config["ACO"]
        self.iterations = int(aco_config.get("iterations", 100))
        self.num_ants = int(aco_config.get("num_ants", 10))
        self.pheromone = float(aco_config.get("pheromone", 1))
        self.evaporation_rate = float(aco_config.get("evaporation_rate", 0.1))

        # SVR parameters.
        svr_config = config["SVR"]
        self.kernel = svr_config.get("kernel", "linear")
        self.epsilon = float(svr_config.get("epsilon", 0.01))
        self.C_min = int(svr_config.get("C_min", 0.01))
        self.C_max = int(svr_config.get("C_max", 100))
        self.gamma_min = int(svr_config.get("gamma_min", 0.01))
        self.gamma_max = int(svr_config.get("gamma_max", 10))

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

        self.parameter_values = [
            (C, gamma)
            for C in np.logspace(self.C_min, self.C_max, 2 * (self.C_max - self.C_min))
            for gamma in np.logspace(
                self.gamma_min, self.gamma_max, 2 * (self.gamma_max - self.gamma_min)
            )
        ]
        self.pheromone_levels = [self.pheromone] * len(self.parameter_values)

    # Define the SVR objective function to be minimized by ACO
    def svr(self, params):
        C, gamma = params
        svr_model = SVR(C=C, epsilon=self.epsilon, gamma=gamma, kernel=self.kernel)
        svr_model.fit(self.X_train, self.Y_train)
        Y_pred = svr_model.predict(self.X_test)

        # You can use any regression metric like Mean Squared Error (MSE) here
        MSE = np.mean((self.Y_test - Y_pred) ** 2)
        return MSE

    def run_optimizer(self):
        for i in range(self.iterations):
            # Initialize the best solution and its error
            best_solution = None
            best_error = float("inf")

            # Ants choose parameter values
            for ant in range(self.num_ants):
                # Use pheromone levels to influence parameter choice
                pheromone_prob = [
                    p / sum(self.pheromone_levels) for p in self.pheromone_levels
                ]
                chosen_index = np.random.choice(
                    len(self.parameter_values), p=pheromone_prob
                )
                chosen_C, chosen_gamma = self.parameter_values[chosen_index]

                # Evaluate the objective function with chosen parameters
                error = self.svr((chosen_C, chosen_gamma))

                # Update the best solution if a better one is found
                if error < best_error:
                    best_solution = (chosen_C, chosen_gamma)
                    best_error = error

            # Update pheromone levels based on the best solution
            for index, (C, gamma) in enumerate(self.parameter_values):
                if (C, gamma) == best_solution:
                    self.pheromone_levels[index] += 1.0 / (1 + best_error)

            # Evaporate pheromone levels to encourage exploration
            self.pheromone_levels = [
                p * (1 - self.evaporation_rate) for p in self.pheromone_levels
            ]

            print(best_solution)

        # Print the best parameters found by ACO
        print("Best Parameters (C, gamma):", best_solution)
        self.best_params = best_solution

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

    print("Initialising ACO...")
    optimizer = ACO(config_file)

    print("Optimizing SVR using ACO...\n")
    optimizer.run_optimizer()

    print("Training SVR using Best Params...")
    optimizer.train_svr()
