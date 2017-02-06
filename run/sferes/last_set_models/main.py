#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To hand search a solution for m and p monkeys

import sys
import os
import numpy as np
from pylab import *
sys.path.append("../src")


from bayesian_1 import bayesian_1
from qlearning_1 import qlearning_1
from mixture_1 import mixture_1
from fusion_1 import fusion_1

parameters = {'length':4,
				'threshold':1.0,
				'noise':0.01,
				'sigma':1.0}
parameters = {'alpha':0.5,
				'beta':10.0,
				'sigma':1.0,
				'kappa':0.5,
				'shift':0.1}
parameters = {'length':5,
				'threshold':1.0,
				'noise':0.1,
				'alpha':0.5,
				'beta':10,
				'sigma':1.0,
				'weight':0.5,
				'kappa':0.5,
				'shift':0.5}
parameters = {"beta":10.0, # temperature for final decision                            
              'alpha':0.5,
              "length":4,
              "threshold":10.0, # sigmoide parameter
              "noise":0.1,
              "gain":0.01,
              "sigma":1.0,
              "gamma":1.0, # temperature for entropy from qlearning soft-max
              "kappa":1.0,
              "shift":1.0}				


list_of_problems = np.random.randint(4, size = 10)

model = fusion_1()
model.test_call(list_of_problems, parameters)






