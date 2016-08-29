#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from pylab import *
import sys, os
sys.path.append("../../src")
from Models import *
from Sferes import pareto
import cPickle as pickle

with open("SFERES_11_best_parameters.pickle", 'rb') as f:
	data = pickle.load(f)


front = pareto()

for o in data.iterkeys():
	print o
	parameters = data[o]['m']['metaf']



	model = MetaFSelection()
	model.analysis_call(front.monkeys['r'], front.rt_reg_monkeys['r'], parameters)
	




	figure(figsize = (15,7))
	rcParams['ytick.labelsize'] = 8
	rcParams['xtick.labelsize'] = 8

	t = 0
	for i in xrange(1,10,2):
		for j in xrange(2):
			subplot(5,2,i+j)
			plot(model.meta_list[:,t,j])
			plot(np.convolve(model.meta_list[:,t,j], np.ones(300)/300, mode = 'full'), color = 'black', linewidth = 2)
			ylim(0,2)
			if j == 0:
				ylabel("Search "+str(t+1))
			elif j == 1:
				ylabel("Repeat "+str(t+1))
			xlabel("Trial")
		t+=1
	

	savefig("SFERES_11_HMETA_"+o+"_singe_m.pdf")





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