#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from pylab import *
import sys, os
sys.path.append("../../src")
from Models import *
from Sferes import pareto
import cPickle as pickle

with open("SFERES_1_best_parameters.pickle", 'rb') as f:
	data = pickle.load(f)



front = pareto()


figure()
for i in xrange(1, 6):
	subplot(2,3,i)
	index = np.where(front.rt_reg_monkeys['m'][:,0] == i)[0]
	plot(np.arange(0,i), front.rt_reg_monkeys['m'][index[0:i],1], 'o-')
	plot(np.arange(i, i+len(front.rt_reg_monkeys['m'][index[i:],1])), front.rt_reg_monkeys['m'][index[i:],1], '*-')
show()



sys.exit()
parameters = data['tche']['m']['fusion']
parameters['gain'] = 1.0

model = FSelection()

fit = model.sferes_call(front.monkeys['m'], front.rt_reg_monkeys['m'], parameters)

print fit[0], fit[1]

figure()

for i in xrange(1, 6):	
	subplot(2,3,i)
	plot(model.rt_model[model.mean_rt[:,0] == i], 'o-')
	plot(model.mean_rt[model.mean_rt[:,0] == i,1], '*--', color = 'black')

show()