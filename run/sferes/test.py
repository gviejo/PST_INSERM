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

model = CSelection()
fit = model.sferes_call(front.monkeys['s'], front.rt_reg_monkeys['s'], parameters)
print fit[0], fit[1]



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