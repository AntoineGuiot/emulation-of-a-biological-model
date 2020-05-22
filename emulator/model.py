import json
import os
import pandas as pd
import docker as docker
import numpy as np

file_path = os.path.dirname(os.path.realpath(__file__))


class Model:

    def __init__(
            self,
            name, function, descriptors, y_values, time
    ):

        self.function = function

        self.name = name

        self.descriptors = descriptors

        self.y_values = y_values
        self.time = time

    def set_vpop_config(self, vpop):
        vpop_config_file_path = f"{file_path}/inputs/Vpop.json"

        ## set vpop config

        y_0_names = [f"{name}_0" for name in self.y_values]

        list_parameters = [
            dict(zip(self.descriptors + y_0_names, sample)) for sample in vpop.patients.values]

        vpop_config = [
            {
                "patientIndex": i,
                "patientAttributes": [[str(key), param[key]] for key in param],
            }
            for i, param in enumerate(list_parameters)
        ]

        with open(vpop_config_file_path, "w") as outfile:
            json.dump(vpop_config, outfile)

    def get_simulation_outputs(self, vpop):

        output_data_frame = pd.DataFrame()
        patient_discarded_values = pd.DataFrame()
        for i in range(vpop.number_of_point):
            row_data_frame = pd.DataFrame()
            row_patient_discarded_values = pd.DataFrame()
            try:
                with open(f"{file_path}/outputs/Patient-{i}.json") as json_file:
                    patient_res = json.load(json_file)

                with open(f"{file_path}/inputs/Vpop.json") as json_file:
                    patient = json.load(json_file)[i]

                for j, param in enumerate(self.x_parameters):
                    row_data_frame[param] = [patient["patientAttributes"][j][1]]

                for j, param in enumerate(self.y_values):
                    # try:  # exclude patients with infinite outputs values
                    row_data_frame[param] = [patient_res[f"DefaultComp.{param}"]["resVals"]]
                    if np.max(np.abs([patient_res[f"DefaultComp.{param}"]["resVals"]])) > 10e5 or np.min(
                            [patient_res[f"DefaultComp.{param}"]["resVals"]]) < 0:
                        print(f'Warning: patient number {i} got output values beyond acceptable range')
                        row_data_frame = pd.DataFrame()

                        # we save parameters were values are infinite
                        for j, param in enumerate(self.x_parameters):
                            row_patient_discarded_values[param] = [patient["patientAttributes"][j][1]]
                        patient_discarded_values = pd.concat(
                            [patient_discarded_values, row_patient_discarded_values],
                            axis=0).dropna()

            except FileNotFoundError:
                print(f'Warning: missing output file for patient number {i}. Patient discarded')
                pass
            output_data_frame = pd.concat([output_data_frame, row_data_frame], axis=0).dropna()
        patient_discarded_values.to_csv("benchmark/discarded_patient.csv")
        return output_data_frame

    def save_simulation_outputs(self, simulation_outputs, vpop):
        simulation_outputs.to_json(
            f"{file_path}/outputs/saved_samples_{self.type}_{vpop.sampling_method}_{vpop.number_of_point}_{vpop.min_}_{vpop.max_}.json",
            orient='split', index=False
        )

    def run(self, vpop, solver):
        # set vpop config
        self.set_vpop_config(vpop)

        try:
            simulation_outputs = pd.read_json(
                f"{file_path}/outputs/saved_samples_{self.type}_{vpop.sampling_method}_{vpop.number_of_point}_{vpop.min_}_{vpop.max_}.json",
                orient='split')
            print('Sampling already exist')
        except:
            # run docker image on vpop config

            for i in range(vpop.number_of_point):
                with open(f"{file_path}/inputs/Vpop.json") as json_file:
                    patient = json.load(json_file)[i]
                    y_0_names = [f"{name}_0" for name in self.y_values]
                    y_0 = patient[y_0_names]
                    x_parameters = patient[self.descriptors]

                    solution = solver.solve(self.function, y_0, tuple(x_parameters))
                patient_res = {}
                for j, param in enumerate(self.y_values):
                    patient_res[f"DefaultComp.{param}"]["resVals"] = solution[:, i]

                with open(f"{file_path}/outputs/Patient-{i}.json", "w") as outfile:
                    json.dump(patient_res, outfile)

            # get outputs simulations
            simulation_outputs = self.get_simulation_outputs(vpop)
            self.save_simulation_outputs(simulation_outputs, vpop)

        return simulation_outputs
