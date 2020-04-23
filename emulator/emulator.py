import json
import os
import pickle
import pyDOE
import sobol_seq
import time
import numpy as np
import docker as docker
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, LSTM, Input, concatenate, Flatten, Reshape
from sklearn.metrics import mean_squared_error
import copy

file_path = os.path.dirname(os.path.realpath(__file__))


class Emulator:
    def __init__(
            self,
            name,
            x_parameters=["alpha", "beta", "delta", "gamma", "DefaultComp.predator", "DefaultComp.prey"],
            t_series_names=["prey", "predator"],
            normalize=True, drop_outliers=True):

        self.name = name
        self.x_parameters = x_parameters
        self.t_series_names = t_series_names
        self.normalize = normalize
        self.drop_outliers = drop_outliers

    def format_input(self, input_parameters):
        x = input_parameters
        # reshape x_parameters in (None, 6, 31)
        x_reshaped = np.repeat(x[:, :, np.newaxis], self.number_time_step, axis=2)
        x_reshaped = np.array([np.array(x_reshaped[i].T) for i in range(len(x_reshaped))])
        return x_reshaped, x

    def format_data(self, simulation_outputs):
        simulation_outputs_copy = copy.deepcopy(simulation_outputs)
        # create time size
        self.number_time_step = len(simulation_outputs_copy[self.t_series_names[0]].values[0])

        if self.normalize == True:
            # create min and max
            for name in self.t_series_names:
                signal_list_as_array = np.vstack([list_ for list_ in simulation_outputs_copy[name].values])
                simulation_outputs_copy['max_' + name] = np.max(signal_list_as_array, axis=1)
                # we will predict the log max
                simulation_outputs_copy['min_' + name] = np.min(signal_list_as_array, axis=1)

            # normalize
            for name in self.t_series_names:
                # signal_list_as_array = np.vstack([list_ for list_ in simulation_outputs[name].values])
                for i in range(len(simulation_outputs_copy)):
                    y = np.array(simulation_outputs_copy[str(name)].iloc[i])
                    y_max = simulation_outputs_copy['max_' + name].iloc[i]
                    y_min = simulation_outputs_copy['min_' + name].iloc[i]
                    simulation_outputs_copy[str(name)].iloc[i] = (y - y_min) / (
                            y_max - y_min)  # TODO move this line at line 76

        # drop outliers where max is up to quantile(0.99)
        if self.drop_outliers == True:
            for name in self.t_series_names:
                simulation_outputs_copy = simulation_outputs_copy[
                    (simulation_outputs_copy['max_' + str(name)] < simulation_outputs_copy['max_' + str(name)].quantile(
                        0.99))]

        # Set usefull values (x_min_max, x_signal, y_min_max, y_signal

        input_parameters = simulation_outputs_copy[self.x_parameters].values
        x_signal, x_min_max = self.format_input(input_parameters)

        if self.normalize == True:
            y_min_max = simulation_outputs_copy[
                [f"max_{name}" for name in self.t_series_names] + [f"min_{name}" for name in
                                                                   self.t_series_names]].values
        else:
            y_min_max = None
            x_min_max = None

        # reshape t_series_values in (None, 2, 31)

        y_signal = simulation_outputs_copy[self.t_series_names].values
        y_signal = np.array(
            [np.array(([np.array(y_signal[i][0]), np.array(y_signal[i][1])])) for i in range(len(y_signal))])
        return x_signal, y_signal, x_min_max, y_min_max

    def train(self,
              # train data
              intput_signal_model_train,
              output_signal_model_train,
              input_min_max_model_train=None,
              output_min_max_model_train=None,
              test_data=True,
              test_data_signal=None,
              test_data_min_max=None
              ):
        number_descriptors = len(self.x_parameters)
        number_time_step = np.shape(intput_signal_model_train)[1]

        # model to predict signal
        inputA = Input(shape=(number_time_step, number_descriptors))
        x = Dense((128), activation="relu")(inputA)
        x = LSTM(128, return_sequences=True, input_shape=(number_time_step, number_descriptors))(x)
        x = Dense(2)(x)
        x = Reshape((2, number_time_step))(x)
        model_signal = Model(inputs=[inputA], outputs=x)
        model_signal.compile(loss='MSE', optimizer='adam')
        # model_signal.summary()  # show the summary of this model in logs

        if test_data == True:
            validation_data_min_max = test_data_min_max
            validation_data_signal = test_data_signal
        else:
            validation_data_min_max = None
            validation_data_signal = None

        if self.normalize == True:
            # model to predict min and max
            # model_min_max.summary() # show the summary of this model in logs
            # Fit
            prediction_sum = np.array([0, 0, 0, 0])
            i = 0
            while np.sum(prediction_sum < 30) > 0 and i < 10:
                inputB = Input(shape=(number_descriptors,))
                y = Dense(128, activation='relu')(inputB)
                y = Dense(32, activation='relu')(y)
                y = Dense(8, activation='relu')(y)
                y = Dense(4, activation='relu')(y)
                print(f'training min _max : {i}')
                i += 1
                model_min_max = Model(inputs=[inputB], outputs=y)
                model_min_max.compile(loss='MSE', optimizer='adam')
                history_min_max = model_min_max.fit(input_min_max_model_train,
                                                    output_min_max_model_train,
                                                    epochs=200,
                                                    batch_size=256,
                                                    validation_data=validation_data_min_max,
                                                    verbose=1)

                self.emulator_min_max = model_min_max
                print(self.emulator_min_max.predict(validation_data_min_max[0]))
                print(self.emulator_min_max.predict(validation_data_min_max[0]).shape)
                prediction_sum = np.sum(self.emulator_min_max.predict(validation_data_min_max[0]), axis=0)
                print(prediction_sum)

        history_signal = model_signal.fit(intput_signal_model_train,
                                          output_signal_model_train,
                                          epochs=200,
                                          batch_size=256,
                                          validation_data=validation_data_signal,
                                          verbose=1)

        self.emulator_signal = model_signal

    def predict(self, input_params):
        final_signal = []
        input_signal_model, input_min_max_model = self.format_input(input_params)
        if self.normalize == True:
            signal, min_max = self.emulator_signal.predict(input_signal_model), self.emulator_min_max.predict(
                input_min_max_model)
            print(min_max)
            max_ = min_max[:, 0:int(min_max.shape[1] / 2)]
            min_ = min_max[:, int(min_max.shape[1] / 2):]

            for i in range(signal.shape[1]):
                final_signal.append(
                    signal[:, i] * (max_[:, i].reshape(len(signal), 1) - min_[:, i].reshape(len(signal), 1)) + min_[:,
                                                                                                               i].reshape(
                        len(signal), 1)
                )
            prediction = np.stack(final_signal, axis=1)

            print('prediction shape : ', prediction.shape)
        else:
            prediction = self.emulator_signal.predict(input_signal_model)
        return prediction

    def mse(self,
            prediction,
            true):

        # y_signal = true #simulation_outputs_copy[self.t_series_names].values
        true = np.array(  # TODO can't handle more than 2 time serie
            [np.array(([np.array(true[i][0]), np.array(true[i][1])])) for i in range(len(true))])

        row_mse_df = pd.DataFrame()
        row_mse_df = row_mse_df.astype('object')
        for (i, name) in enumerate(self.t_series_names):
            row_mse_df[f'mse_{name}'] = [mean_squared_error(true[:, i], prediction[:, i])]
            residus = true[:, i] - prediction[:, i]
            row_mse_df[f'residus_{name}'] = [list(np.concatenate(residus, axis=0))]
            row_mse_df[f'predict_{name}'] = [list(prediction[:, i])]
            row_mse_df[f'true_{name}'] = [list(true[:, i])]
            # we will compute the mse for max_prey ,min_prey ,max_predator, min_predator
            # with  min_max_true = [max_prey, max_predator, min_prey, min_predator] (same for min_max_prediction)

            # max_true = np.exp(min_max_true[:, i].reshape(len(min_max_true), 1))
            # min_true = min_max_true[:, i + len(self.t_series_names)].reshape(len(min_max_true), 1)

            # prediction
            # max_prediction = np.exp(min_max_prediction[:, i].reshape(len(min_max_prediction), 1))
            # min_prediction = min_max_prediction[:, i + len(self.t_series_names)].reshape(len(min_max_prediction), 1)

            # entire time series
            # entire_time_series_true = signal_true[:, i] * (max_true - min_true) + min_true
            # entire_time_series_prediction = signal_prediction[:, i] * (max_prediction - min_prediction) + min_prediction

            # row_mse_df[f'mse_tot_{name}'] = [mean_squared_error(entire_time_series_true, entire_time_series_prediction)]
        return row_mse_df

        # def test(self, test_df):

    #    y_pred = self.model.predict(test_df[self.x_parameters].values)
    #    y_true = np.concatenate(
    #        (np.array(list(test_df['prey'].values)), np.array(list(test_df['predator'].values))),
    #        axis=1)
    #    list_error = ((y_true - y_pred) ** 2).flatten()
    #    self.median = np.median(list_error)
    #    self.q_75 = np.quantile(list_error, 0.75)
    #    self.q_25 = np.quantile(list_error, 0.25)
    #    self.q_90 = np.quantile(list_error, 0.90)
    #    self.q_10 = np.quantile(list_error, 0.10)
    #    self.mean = np.mean(list_error)
    #    self.square_err = (y_true - y_pred) ** 2
    #    self.std = np.std((y_true - y_pred) ** 2)
    #    self.score = mse(y_true, y_pred)

    def save_model(self):
        with open(
                f"{file_path}/outputs/model_{self.number_of_point}_{self.sampling_method}.pkl",
                "wb",
        ) as f:
            pickle.dump(self, f)
