import configparser
import pickle
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils import load_data


class CS:
    def __init__(self, config_file):
        # Parse the config file.
        config = configparser.ConfigParser()
        config.read(config_file)

        # CS parameters.
        cs_config = config["CS"]
        self.iterations = int(cs_config.get("iterations", 100))
        self.population_size = int(cs_config.get("population_size", 20))
        self.pa = float(cs_config.get("pa", 0.25))
        self.dimension = int(cs_config.get("dimension", 2))

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

        self.population = [
            [
                np.random.uniform(self.C_min, self.C_max),
                np.random.uniform(self.gamma_min, self.gamma_max),
            ]
            for _ in range(self.population_size)
        ]
        print(self.population)

    # Define the SVR objective function to be minimized by CS
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
            # Evaluate fitness for each solution in the population
            fitness_values = [self.svr(solution) for solution in self.population]

            # Sort the population by fitness in ascending order
            sorted_population = [
                x for _, x in sorted(zip(fitness_values, self.population))
            ]

            # Replace a fraction of worst solutions with new solutions (Lévy flights)
            num_replace = int(self.pa * self.population_size)
            replacement_solutions = [
                [
                    np.random.uniform(self.C_min, self.C_max),
                    np.random.uniform(self.gamma_min, self.gamma_max),
                ]
                for _ in range(num_replace)
            ]

            # Perform a random replacement of worst solutions
            for i in range(num_replace):
                index_to_replace = np.random.randint(0, self.population_size - 1)
                sorted_population[index_to_replace] = replacement_solutions[i]

            # Update the population with the new solutions
            population = sorted_population

            # Optionally, you can print the best solution found so far
            best_solution = population[0]
            best_fitness = self.svr(best_solution)

            print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

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

    print("Initialising CS...")
    optimizer = CS(config_file)

    print("Optimizing SVR using CS...\n")
    optimizer.run_optimizer()

    print("Training SVR using Best Params...")
    optimizer.train_svr()
