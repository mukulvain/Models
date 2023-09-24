import configparser
import pickle
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils import load_data


class CCS:
    def __init__(self, config_file):
        # Parse the config file.
        config = configparser.ConfigParser()
        config.read(config_file)

        # CCS parameters.
        ccs_config = config["CCS"]
        self.iterations = int(ccs_config.get("iterations", 100))
        self.population_size = int(ccs_config.get("population_size", 20))
        self.pa = float(ccs_config.get("pa", 0.25))
        self.dimension = int(ccs_config.get("dimension", 2))
        self.chaotic_parameter = float(ccs_config.get("chaos", 0.25))

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

    def chaotic_map(self, x, a):
        return a * x * (1 - x)

    # Define the SVR objective function to be minimized by CCS
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

            # replacement_solutions = [
            #     [
            #         np.random.uniform(self.C_min, self.C_max),
            #         np.random.uniform(self.gamma_min, self.gamma_max),
            #     ]
            #     for _ in range(num_replace)
            # ]
            replacement_solutions = []
            idx = 0
            while len(replacement_solutions) != num_replace:
                # Generate random numbers for chaotic Lévy flights
                chaotic_x = np.random.uniform(0, 1)
                chaotic_a = np.random.uniform(0.1, 0.9)

                # Calculate the Lévy flight steps using the chaotic map
                step_c = (
                    self.chaotic_parameter
                    * self.chaotic_map(chaotic_x, chaotic_a)
                    * np.exp(4)
                )
                step_gamma = (
                    self.chaotic_parameter
                    * self.chaotic_map(chaotic_x, chaotic_a)
                    * np.exp(4)
                )

                # Calculate the new solution
                new_solution = [
                    sorted_population[idx][0] * step_c,
                    sorted_population[idx][1] * step_gamma,
                ]
                if (self.C_min < new_solution[0] and new_solution[0] < self.C_max) and (
                    self.gamma_min < new_solution[1]
                    and new_solution[1] < self.gamma_max
                ):
                    replacement_solutions.append(new_solution)
                    idx += 1

            # Perform a random replacement of worst solutions
            for i in range(num_replace):
                index_to_replace = np.random.randint(0, self.population_size - 1)
                sorted_population[index_to_replace] = replacement_solutions[i]

            # Update the population with the new solutions
            self.population = sorted_population

            # Optionally, you can print the best solution found so far
            best_solution = self.population[0]
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

    print("Initialising CCS...")
    optimizer = CCS(config_file)

    print("Optimizing SVR using CCS...\n")
    optimizer.run_optimizer()

    print("Training SVR using Best Params...")
    optimizer.train_svr()
