import configparser
import pickle
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils import load_data


class MVO:
    def __init__(self, config_file):
        # Parse the config file.
        config = configparser.ConfigParser()
        config.read(config_file)

        # MVO parameters.
        mvo_config = config["MVO"]
        self.iterations = int(mvo_config.get("iterations", 100))
        self.universes = int(mvo_config.get("universes", 20))
        self.entanglement_strength = float(mvo_config.get("entanglement_strength", 0.5))
        self.collapse_factor = float(mvo_config.get("collapse_factor", 0.1))

        # SVR parameters.
        svr_config = config["SVR"]
        self.kernel = svr_config.get("kernel", "linear")
        self.epsilon = float(svr_config.get("epsilon", 0.01))
        self.C_min = float(svr_config.get("C_min", 0.01))
        self.C_max = float(svr_config.get("C_max", 100))
        self.gamma_min = float(svr_config.get("gamma_min", 0.01))
        self.gamma_max = float(svr_config.get("gamma_max", 10))
        self.bounds = [(self.C_min, self.C_max), (self.gamma_min, self.gamma_max)]

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

        self.multiverse = np.random.uniform(
            low=[self.C_min, self.gamma_min],
            high=[self.C_max, self.gamma_max],
            size=(self.universes, 2),
        )

    # Define the SVR objective function to be minimized by MVO
    def svr(self, params):
        C, gamma = params
        svr_model = SVR(C=C, epsilon=self.epsilon, gamma=gamma, kernel=self.kernel)
        svr_model.fit(self.X_train, self.Y_train)
        Y_pred = svr_model.predict(self.X_test)

        # You can use any regression metric like Mean Squared Error (MSE) here
        MSE = np.mean((self.Y_test - Y_pred) ** 2)
        return MSE

    def run_optimizer(self):
        for iteration in range(self.iterations):
            # Evaluate the fitness of each universe
            fitness_values = [self.svr((C, gamma)) for C, gamma in self.multiverse]

            # Quantum entanglement phase
            for i in range(self.universes):
                new_C, new_gamma = 0, 0

                # Ensure that the new positions are within bounds
                while self.C_min > new_C or new_C > self.C_max:
                    new_C = self.multiverse[
                        i, 0
                    ] + self.entanglement_strength * np.random.uniform(-1, 1)

                while self.gamma_min > new_gamma or new_gamma > self.gamma_max:
                    new_gamma = self.multiverse[
                        i, 1
                    ] + self.entanglement_strength * np.random.uniform(-1, 1)

                new_universe = [new_C, new_gamma]
                self.multiverse[i] = new_universe

            # Quantum collapse phase
            collapse_probabilities = np.exp(
                -self.collapse_factor * np.array(fitness_values)
            )
            collapse_probabilities /= np.sum(collapse_probabilities)

            # Select a universe to collapse to based on probabilities
            selected_universe = np.random.choice(
                range(self.universes), p=collapse_probabilities
            )

            # Copy the selected universe's position to all other universes
            for i in range(self.universes):
                self.multiverse[i, :] = self.multiverse[selected_universe, :]

            # Optionally, you can print the best solution found so far in each iteration
            best_solution = self.multiverse[np.argmin(fitness_values)]
            best_fitness = min(fitness_values)

            print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

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

    print("Initialising MVO...")
    optimizer = MVO(config_file)

    print("Optimizing SVR using MVO...\n")
    optimizer.run_optimizer()

    print("Training SVR using Best Params...")
    optimizer.train_svr()
