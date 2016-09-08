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
# model.analysis_call(front.monkeys['p'], front.rt_reg_monkeys['p'], parameters)

model.sferes_call(front.monkeys['p'], front.rt_reg_monkeys['p'], parameters, 'biais')
sum_Log_biais = np.zeros(30)
for i in xrange(len(model.value)):
	sum_Log_biais[model.sari[i,3]] -= model.value[i]

model.sferes_call(front.monkeys['p'], front.rt_reg_monkeys['p'], parameters, 'no biais')
sum_Log_no_biais = np.zeros(30)
for i in xrange(len(model.value)):
	sum_Log_no_biais[model.sari[i,3]] -= model.value[i]



# sum_Log = sum_Log/count

figure()

for i in xrange(5):
	subplot(2,5,i+1)
	plot(sum_Log_biais[front.rt_reg_monkeys['p'][:,0] == i+1], 'o-', label = 'biais')
	plot(sum_Log_no_biais[front.rt_reg_monkeys['p'][:,0] == i+1], 'o-', label = 'no biais')

	legend()
show()





	

