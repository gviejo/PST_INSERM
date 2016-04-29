#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from pylab import *
import sys, os
sys.path.append("../../src")
from Models import *
from Sferes import pareto



parameters = {'alpha': 0.99315900000000001,
 'beta': 4.8539899999999996,
 'kappa': 0.18535099999999999,
 'length': 2.3830840000000002,
 'noise': 0.095417500000000002,
 'shift': 0.63659500000000002,
 'sigma': 12.71744,
 'threshold': 0.49407347000000001,
 'weight': 0.091949400000000001}


front = pareto()

model = CSelection()

fit = model.sferes_call(front.monkeys['p'], front.rt_reg_monkeys['p'], parameters)

print fit[0], fit[1]
