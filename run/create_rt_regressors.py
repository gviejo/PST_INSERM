#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CReate the regressors of the rt
# creaqte for each monkeys
# name_rt_reg.txt
# Nb d'erreurs en search | mean | sem


import numpy as np
from pylab import *
import cPickle as pickle
from scipy.stats import sem

monkeys =   { 'g':12701,
              'm':34752,
              'p':27692,
              'r':11634,
              's':13348 }

data = {}
for s in monkeys.keys():
	with open("../data/data_pickle/"+s+".pickle", 'rb') as f:
		data[s] = pickle.load(f)


rt = {}
rtmean = {}

for s in monkeys.keys():
	rt[s] = dict({'search':{},'repeat':{}})
	rtmean[s] = dict({'search':{},'repeat':{}})
	max_N_search = 0
	max_N_repeat = 0

	problem = data[s][0,4]
	search = [data[s][0,8]]
	repeat = []
	for t in xrange(1, len(data[s])):		
		if data[s][t,4] == problem: # same problem
			if data[s][t,2] == 0.0: # search trial
				search.append(data[s][t,8]) # append rt of search trial
			elif data[s][t,2] == 1.0: # repeat trial
				repeat.append(data[s][t,8]) # append rt of repeat trial
		else: # new problem
			nb_incorrect = len(search) # number of search trial before correct
			if nb_incorrect in rt[s]['search'].keys(): # append search trial in corresponding error length
				rt[s]['search'][nb_incorrect].append(np.array(search))
				rt[s]['repeat'][nb_incorrect].append(np.array(repeat))
			else:
				rt[s]['search'][nb_incorrect] = [np.array(search)]
				rt[s]['repeat'][nb_incorrect] = [np.array(repeat)]
				
			problem = data[s][t,4]  # number of new problem
			search = [data[s][t,8]] 
			repeat = [] 

	t = 'search'
	for l in rt[s][t].keys():
		rt[s][t][l] = np.array(rt[s][t][l])
		rtmean[s][t][l] = np.zeros((2,l))
		rtmean[s][t][l][0] = rt[s][t][l].mean(0)
		rtmean[s][t][l][1] = sem(rt[s][t][l])

	t = 'repeat'
	for l in rt[s][t].keys():
		rt[s][t][l] = np.array(rt[s][t][l])
		meann = np.zeros((2,7))
		for i in xrange(7):
			tmp = []
			for j in xrange(len(rt[s][t][l])):
				if len(rt[s][t][l][j]) > i:
					tmp.append(rt[s][t][l][j][i])
			meann[0,i] = np.mean(tmp)
			meann[1,i] = sem(tmp)
				
		rtmean[s][t][l] = meann

# Write file
for s in monkeys.keys():
	tmp = []
	tmp2 = []
	indice = []
	for i in xrange(1,7):
		tmp.append(np.hstack((rtmean[s]['search'][i][0],rtmean[s]['repeat'][i][0])))
		tmp2.append(np.hstack((rtmean[s]['search'][i][1],rtmean[s]['repeat'][i][1])))
		indice.append(np.ones(i+7)*i)
	tmp = np.concatenate(tmp)
	tmp2 = np.concatenate(tmp2)
	tmp2[np.isnan(tmp2)] = 0.0
	indice = np.concatenate(indice)
	reg = np.vstack((indice, tmp, tmp2)).transpose()
	np.savetxt("../data/data_txt/"+s+"_rt_reg.txt", reg)
