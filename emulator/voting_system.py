import numpy as np

from emulator import Emulator


class VotingSystem:

    def __init__(self, n_emulator, simulation, x_parameters, t_series_names, normalize,
                 drop_outliers):
        self.n_emulator = n_emulator
        self.simulation = simulation
        self.x_parameters = x_parameters
        self.t_series_names = t_series_names
        self.normalize = normalize
        self.drop_outliers = drop_outliers
        self.formater_emulator = Emulator(f'Emulator_formater', self.x_parameters, self.t_series_names, self.normalize,
                                          self.drop_outliers)
        self.x_signal, self.y_signal, self.x_min_max, self.y_min_max = self.formater_emulator.format_data(
            self.simulation)

    def start_training(self):
        self.emulators = dict()
        number_of_points = np.linspace(len(self.x_signal) / 6, len(self.x_signal), self.n_emulator).astype(int)

        number_of_rows = self.x_signal.shape[0]

        for i in range(self.n_emulator):
            print(number_of_points[i])
            print(number_of_rows)
            random_indices = np.random.choice(number_of_rows, size=number_of_points[i], replace=False)
            x_signal = self.x_signal[random_indices, :]
            y_signal = self.y_signal[random_indices, :]
            if self.x_min_max != None:
                x_min_max = self.x_min_max[random_indices, :]
                y_min_max = self.y_min_max[random_indices, :]
            else:
                x_min_max = None
                y_min_max = None

            print(f'training on emulator n° {i} : started ')
            # TODO dev other multi training methods

            emulator = Emulator(f'Emulator_{i}', self.x_parameters, self.t_series_names, self.normalize,
                                self.drop_outliers)
            emulator.train(x_signal, y_signal, x_min_max, y_min_max)
            self.emulators[emulator.name] = emulator
            print(f'training on emulator n° {i} : done ')

    def predict(self, input_params):
        prediction = dict()
        for emulator_name in list(self.emulators.keys()):
            prediction[emulator_name] = self.emulators[emulator_name].predict(input_params)
        return prediction
