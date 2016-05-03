#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from pylab import *
import sys, os
sys.path.append("../../src")
from Models import *
from Sferes import pareto
import cPickle as pickle

with open("SFERES_1_only_fusion_best_parameters.pickle", 'rb') as f:
	data = pickle.load(f)


front = pareto()
parameters = data['tche']['m']['fusion']
parameters['length'] = 3.0
parameters['gain'] = 1.0
parameters['sigma'] = 0.001
print parameters
model = FSelection()
fit = model.sferes_call(front.monkeys['m'], front.rt_reg_monkeys['m'], parameters)
print fit[0], fit[1]

figure()
for i in xrange(1, 6):
	subplot(2,3,i)
	index = np.where(front.rt_reg_monkeys['m'][:,0] == i)[0]
	plot(np.arange(0,i), front.rt_reg_monkeys['m'][index[0:i],1], 'o-', color = 'black')
	plot(np.arange(i, i+len(front.rt_reg_monkeys['m'][index[i:],1])), front.rt_reg_monkeys['m'][index[i:],1], '*-', color='black')

	plot(np.arange(0,i), model.rt_model[index[0:i]], 'o-')
	plot(np.arange(i, i+len(model.rt_model[index[i:]])), model.rt_model[index[i:]], '*-')


show()