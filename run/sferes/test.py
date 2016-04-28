
import numpy as np
from pylab import *
import sys, os
sys.path.append("../../src")
from Models import FSelection
from Sferes import pareto



parameters = {'alpha': 0.024641799999999998,
 'beta': 64.131900000000002,
 'gain': 7443.3700025566304,
 'gamma': 20.218499999999999,
 'kappa': 0.23211000000000001,
 'length': 1.0,
 'noise': 0.012208,
 'shift': 0.064492900000000006,
 'sigma': 0.0,
 'threshold': 430.01800569981998}

front = pareto()

model = FSelection()

fit = model.sferes_call(front.monkeys['r'], front.rt_reg_monkeys['r'], parameters)

print fit[0], fit[1]
