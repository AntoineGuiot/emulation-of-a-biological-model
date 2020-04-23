import numpy as np
import pandas as pd
import time

from emulator import Emulator
from model import Model
from vpop import Vpop


def benchmark(number_of_iter, min_samples, max_samples, type='scorer'):
    number_of_point = np.linspace(min_samples, max_samples, number_of_iter)
    benchmark_data_frame = pd.DataFrame()
    descriptors = ["alpha", "beta", "delta", "gamma", "DefaultComp.predator", "DefaultComp.prey"]
    # create vpop testing
    vpop_test = Vpop(f'vpop testing', descriptors,
                     # fixed_parameters=dict({'pred_0': 1, 'prey_0': 1}),
                     number_of_point=500, min_=0.1, max_=1, sampling_method='random')
    vpop_test.sample()

    # create model
    model = Model('model', type=type)

    testing_data_frame = model.run(vpop_test)

    emulator = Emulator('test formater', x_parameters=descriptors, normalize=False, drop_outliers=False)
    intput_signal_model_test, output_signal_model_test, input_min_max_model_test, output_min_max_model_test = emulator.format_data(
        testing_data_frame)
    test_data_signal = (intput_signal_model_test, output_signal_model_test)
    test_data_min_max = (input_min_max_model_test, output_min_max_model_test)

    print(vpop_test.patients)
    for i in range(number_of_iter):
        print(f'iteration n°{i} : sampling...')

        # create training vpop
        vpop_train = Vpop(f'vpop n°{i}', descriptors=descriptors,
                          # fixed_parameters=dict({'pred_0': 1, 'prey_0': 1}),
                          number_of_point=int(number_of_point[i]), min_=0.1, max_=1)
        vpop_train.sample()

        # run model on vpop
        training_data_frame = model.run(vpop_train)

        # create emulator
        emulator = Emulator(f'emulator_n°{i}', x_parameters=descriptors, normalize=False, drop_outliers=False)
        print(f'iteration n°{i} : sampling done, formating...')
        # format input
        intput_signal_model_train, output_signal_model_train, input_min_max_model_train, output_min_max_model_train = emulator.format_data(
            training_data_frame)

        print(f'iteration n°{i} : training...')



        emulator.train(intput_signal_model_train,  # train
                       output_signal_model_train,
                       input_min_max_model_train,
                       output_min_max_model_train,
                       test_data=True,
                       test_data_signal=test_data_signal,
                       test_data_min_max=test_data_min_max)

        prediction = emulator.predict(input_params=testing_data_frame[descriptors].values)
        true = testing_data_frame[emulator.t_series_names].values

        raw_mse = emulator.mse(prediction,
                               true=true)
        raw_mse['training_set_size'] = vpop_train.number_of_point
        benchmark_data_frame = pd.concat([benchmark_data_frame, raw_mse], axis=0)

        benchmark_data_frame.to_json(
            f"benchmark/benchmark_scorer.json",
            orient='split', index=False
        )

    return benchmark_data_frame


if __name__ == "__main__":
    t1 = time.time()
    benchmark(8, 500, 3000)
    t2 = time.time()
    print(t2 - t1)
