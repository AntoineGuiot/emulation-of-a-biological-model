import pandas as pd

import sampling as sampling
import pandas as pd
import numpy as np
from scipy.stats.distributions import expon


class Vpop:

    def __init__(
            self,
            name,
            descriptors,
            sampling_method="lhs",
            number_of_point=100,
            fixed_parameters=None,  # dict({'parameter': value,} **)
            min_=0,
            max_=1,
            distribution=None):

        self.name = name
        self.descriptors = descriptors
        self.sampling_method = sampling_method
        self.number_of_point = number_of_point
        self.fixed_parameters = fixed_parameters
        self.min_ = min_
        self.max_ = max_
        self.distribution = distribution

    def sample(self):
        number_of_parameter = len(self.descriptors)

        if self.sampling_method == "lhs":
            variable_param_values = sampling.lhs_sampling(
                self.number_of_point, number_of_parameter
            )

        if self.sampling_method == "sobol":
            variable_param_values = sampling.sobol_seq_sampling(
                self.number_of_point, number_of_parameter
            )

        if self.sampling_method == "random":
            variable_param_values = sampling.random_sampling(
                self.number_of_point, number_of_parameter
            )
        variable_param = pd.DataFrame(variable_param_values, columns=self.descriptors)

        if self.distribution != None:
            variable_param = expon(scale=0.5).ppf(variable_param)

        variable_param = (self.max_ - self.min_) * variable_param + self.min_
        variable_param = pd.DataFrame(variable_param, columns=self.descriptors)
        if self.fixed_parameters != None:
            fixed_param = pd.DataFrame(
                np.array([list(self.fixed_parameters.values())]).repeat(self.number_of_point, axis=0),
                columns=list(self.fixed_parameters.keys()))
        else:
            fixed_param = pd.DataFrame()

        self.patients = pd.concat([variable_param, fixed_param], axis=1)
