#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from pylab import *
import sys, os
sys.path.append("../../src")
from Models import *
from Sferes import pareto
import cPickle as pickle

with open("../sferes/SFERES_14_best_parameters.pickle", 'rb') as f:
	data = pickle.load(f)

front = pareto()
parameters = data['tche']['p']['sweeping']





model = Sweeping()
model.analysis_call(front.monkeys['p'], front.rt_reg_monkeys['p'], parameters)
	

