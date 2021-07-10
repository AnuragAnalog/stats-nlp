#!/usr/bin/python3

# T-Test
import numpy as np

def t_test(mean, element, variance, size):
    return (element-mean)/np.sqrt(variance/size)

# Chi Square Test
def chi_sq_test(observed, expected):
    return np.sum((observed-expected)**2/expected)

# Log Likelihood Test
def log_likelihood():
    return np.log()

# Mutual Information Test
def information(Px, Py, Pxy):
    return np.log(Pxy/(Px*Py))