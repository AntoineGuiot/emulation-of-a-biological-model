import json
import os
import pandas as pd
import docker as docker
import numpy as np

file_path = os.path.dirname(os.path.realpath(__file__))


class Model:

    def __init__(
            self,
            name, type='solver'
    ):

        self.type = type

        if self.type == 'solver':
            x_parameters = ["alpha", "beta", "delta", "gamma", "DefaultComp.predator", "DefaultComp.prey"]
            y_values = ["prey", "predator"]
            docker_image = "registry.novadiscovery.net/students/model-emulator/vpop-solver:v1"
            docker_volumes = [f"{file_path}/inputs:/inputs", f"{file_path}/outputs:/outputs"]
            docker_command = '/inputs/SolverConfig.json \
                                  --loc="/Inputs/RawModel=/inputs/RawModel.json" \
                                  --loc="/Vpop/Outputs/Patients=/inputs/Vpop.json" \
                                  --loc="/Vpop/Outputs/Results/Patient=/outputs/Patient-{patientNumber}.json" \
                                  --var simulation_id=0'

        if self.type == 'scorer':
            x_parameters = ["alpha", "beta", "delta", "gamma", "DefaultComp.predator", "DefaultComp.prey"]
            y_values = ["prey", "predator"]
            docker_image = "registry.novadiscovery.net/students/model-emulator/vpop-scorer:v1"
            docker_volumes = [f'{file_path}/vpop-scorer-inputs:/inputs', f'{file_path}/outputs:/outputs']
            docker_command = '/inputs/SolverConfig.json'

        self.name = name

        self.docker_image = docker_image
        self.docker_volumes = docker_volumes
        self.docker_command = docker_command
        self.docker_client = docker.from_env()

        self.x_parameters = x_parameters
        self.y_values = y_values

    def set_vpop_config(self, vpop):
        if self.type == 'solver':
            vpop_config_file_path = f"{file_path}/inputs/Vpop.json"

        if self.type == 'scorer':
            vpop_config_file_path = f"{file_path}/vpop-scorer-inputs/Vpop.json"

        ## set vpop config

        list_parameters = [
            dict(zip(self.x_parameters, sample)) for sample in vpop.patients.values]

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

        if self.type == 'solver':
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

        if self.type == 'scorer':
            output_data_frame = pd.DataFrame()
            patient_discarded_values = pd.DataFrame()
            with open(f"{file_path}/outputs/Evaluation.json") as json_file:
                evaluations = json.load(json_file)

            with open(f"{file_path}/vpop-scorer-inputs/Vpop.json") as json_file:
                patients = json.load(json_file)

            for i in range(vpop.number_of_point):
                row_data_frame = pd.DataFrame()
                row_patient_discarded_values = pd.DataFrame()
                d = evaluations['exploPatients'][i]['scores']

                v = {k: [dic[k] for dic in d] for k in d[0]}['value']
                for j, param in enumerate(self.x_parameters):
                    row_data_frame[param] = [patients[i]["patientAttributes"][j][1]]

                for j, param in enumerate(self.y_values):
                    k = len(v) / len(self.y_values)
                    k = int(k)
                    row_data_frame[param] = [v[j * k:(j + 1) * k]]

                if np.max(np.abs(v)) > 10e5 or np.min(v) < 0:
                    print(f'Warning: patient number {i} got output values beyond acceptable range')
                    row_data_frame = pd.DataFrame()

                    # we save parameters were values are infinite
                    for j, param in enumerate(self.x_parameters):
                        row_patient_discarded_values[param] = [patients[i]["patientAttributes"][j][1]]
                    patient_discarded_values = pd.concat(
                        [patient_discarded_values, row_patient_discarded_values],
                        axis=0).dropna()

                output_data_frame = pd.concat([output_data_frame, row_data_frame], axis=0).dropna()
            patient_discarded_values.to_csv("benchmark/discarded_patient.csv")
            return output_data_frame

    def save_simulation_outputs(self, simulation_outputs, vpop):
        simulation_outputs.to_json(
            f"{file_path}/outputs/saved_samples_{self.type}_{vpop.sampling_method}_{vpop.number_of_point}_{vpop.min_}_{vpop.max_}.json",
            orient='split', index=False
        )

    def run(self, vpop):
        # set vpop config
        self.set_vpop_config(vpop)

        try:
            simulation_outputs = pd.read_json(
                f"{file_path}/outputs/saved_samples_{self.type}_{vpop.sampling_method}_{vpop.number_of_point}_{vpop.min_}_{vpop.max_}.json",
                orient='split')
            print('Sampling already exist')
        except:
            # run docker image on vpop config
            self.docker_client.containers.run(
                image=self.docker_image,
                command=self.docker_command,
                volumes=self.docker_volumes,
                auto_remove=True)

            # get outputs simulations
            simulation_outputs = self.get_simulation_outputs(vpop)
            self.save_simulation_outputs(simulation_outputs, vpop)

        return simulation_outputs
