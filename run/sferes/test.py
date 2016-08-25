#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from pylab import *
import sys, os
sys.path.append("../../src")
from Models import *
from Sferes import pareto
import cPickle as pickle

with open("SFERES_7_best_parameters.pickle", 'rb') as f:
	data = pickle.load(f)


front = pareto()
parameters = data['distance']['s']['mixture']

parameters = {'alpha': 0.0016056099999999999,
 'beta': 13.8203,
 'eta': 0.992838,
 'gain': 0.37666199962334801,
 'gamma': 92.144000000000005,
 'kappa': 0.080751299999999998,
 'length': 2.0349550000000001,
 'noise': 0.050000000000000003,
 'shift': 0.87298500000000001,
 'sigma': 19.077280000000002,
 'threshold': 4.8594200000000001}

model = MetaFSelection()
model.analysis_call(front.monkeys['m'], front.rt_reg_monkeys['m'], parameters)
# print fit[0], fit[1]



sys.exit()

figure()
for i in xrange(1, 6):
	subplot(2,3,i)
	index = np.where(front.rt_reg_monkeys['m'][:,0] == i)[0]
	plot(np.arange(0,i), front.rt_reg_monkeys['m'][index[0:i],1], 'o-', color = 'black')
	plot(np.arange(i, i+len(front.rt_reg_monkeys['m'][index[i:],1])), front.rt_reg_monkeys['m'][index[i:],1], '*-', color='black')

	plot(np.arange(0,i), model.rt_model[index[0:i]], 'o-')
	plot(np.arange(i, i+len(model.rt_model[index[i:]])), model.rt_model[index[i:]], '*-')


show()