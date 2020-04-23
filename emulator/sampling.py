import numpy as np
import pyDOE
import sobol_seq


def lhs_sampling(number_of_point, number_of_param, min_=0, max_=1):
    return (max_ - min_) * pyDOE.lhs(number_of_param, samples=number_of_point, criterion="cm") + min_


def sobol_seq_sampling(number_of_point, number_of_param, min_=0, max_=1):
    return (max_ - min_) * sobol_seq.i4_sobol_generate(number_of_param, number_of_point) + min_


def random_sampling(number_of_point, number_of_param, min_=0, max_=1):
    return (max_ - min_) * np.random.random((number_of_point, number_of_param)) + min_


def uniform_sampling(number_of_point, number_of_param, min_=0, max_=1):
    return #TODO
