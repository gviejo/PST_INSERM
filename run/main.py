#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To hand search a solution for m and p monkeys

import sys
import os
import numpy as np
from pylab import *
sys.path.append("../src")

from Models import *
from bayesian_1 import bayesian_1


parameters = {'length':4,
				'threshold':1.0,
				'noise':0.01
				'sigma':1.0}

list_of_problems = np.random.randint(4, 10)

model = bayesian_1()
model.test_call(list_of_problems, parameters)





