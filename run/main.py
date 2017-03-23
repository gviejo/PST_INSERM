#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To hand search a solution for m and p monkeys

import sys
import os
import numpy as np
from pylab import *
sys.path.append("../src")

from Models import *
from fusion_1 import fusion_1


parameters = dict({"beta":1.0, # temperature for final decision                            
							'alpha':0.5,
							"length":4,
							"threshold":2.0, # sigmoide parameter
							"noise":0.0,
							"gain":10.0, # sigmoide parameter 
							"sigma":10.0,
							"gamma":3.0, # temperature for entropy from qlearning soft-max
							"kappa":0.5,
							"shift":0.5})

list_of_problems = np.random.randint(4, 10)

model = fusion_1()
model.test_call(10, list_of_problems, parameters)





