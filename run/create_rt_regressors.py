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
import sys, os


monkeys =   { 'g':12701,
              'm':34752,
              'p':27692,
              'r':11634,
              's':13348 }

data = {}
for s in monkeys.keys():
	with open("../data/data_pickle/"+s+".pickle", 'rb') as f:
		data[s] = pickle.load(f)

nb_repeat = 3

rt = {}
rtmean = {}
new_data = {}

for s in monkeys.keys():
	rt[s] = dict({'search':{},'repeat':{}})
	rtmean[s] = dict({'search':{},'repeat':{}})	
	index = np.ones(len(data[s]))*-1
	keep = np.zeros(len(data[s]))
	problem = data[s][0,4]
	search = [data[s][0,8]]
	repeat = []
	phase = 0.0
	start = 0
	order = np.concatenate([np.ones(v)*(v-nb_repeat) for v in range(1+nb_repeat,5+nb_repeat+1)])
	for t in xrange(1, len(data[s])):		
		phase = data[s][t,2]-data[s][t-1,2] # Si phase == -1, on vient de changer de problem
		if data[s][t,4] == problem and (phase == 0.0 or phase == 1.0): # same problem
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
				
			# set index
			if nb_incorrect <= 5: # on garde ce problem
				if t-start <= nb_repeat: 
					index[start:t] = np.where(order==nb_incorrect)[0][0:t-start]
				else:
					index[start:start+nb_incorrect+nb_repeat] = np.where(order==nb_incorrect)[0]
				keep[start:start+nb_incorrect+nb_repeat] = 1
			start = t
			problem = data[s][t,4]  # number of new problem
			search = [data[s][t,8]] 
			repeat = [] 
	
	# concatenante index for least-square error for ccall
	# keep only problem of interest
	new_data[s] = np.hstack((data[s], np.vstack(index)))[keep == 1.0]

	t = 'search'
	for l in rt[s][t].keys():
		rt[s][t][l] = np.array(rt[s][t][l])
		rtmean[s][t][l] = np.zeros((2,l))
		rtmean[s][t][l][0] = rt[s][t][l].mean(0)
		rtmean[s][t][l][1] = sem(rt[s][t][l])

	t = 'repeat'
	for l in rt[s][t].keys():
		rt[s][t][l] = np.array(rt[s][t][l])
		meann = np.zeros((2,nb_repeat))
		for i in xrange(nb_repeat):
			tmp = []
			for j in xrange(len(rt[s][t][l])):
				if len(rt[s][t][l][j]) > i:
					tmp.append(rt[s][t][l][j][i])
			meann[0,i] = np.mean(tmp)
			meann[1,i] = sem(tmp)
				
		rtmean[s][t][l] = meann
	print s, int(keep.sum())


# Write file
for s in monkeys.keys():
	tmp = []
	tmp2 = []
	indice = []
	for i in xrange(1,6):
		tmp.append(np.hstack((rtmean[s]['search'][i][0],rtmean[s]['repeat'][i][0])))
		tmp2.append(np.hstack((rtmean[s]['search'][i][1],rtmean[s]['repeat'][i][1])))
		indice.append(np.ones(i+nb_repeat)*i)
	tmp = np.concatenate(tmp)
	tmp2 = np.concatenate(tmp2)
	tmp2[np.isnan(tmp2)] = 0.0
	indice = np.concatenate(indice)
	reg = np.vstack((indice, tmp, tmp2)).transpose()
	
	os.system("mkdir ../data/data_txt_"+str(nb_repeat)+"_repeat")
	np.savetxt("../data/data_txt_"+str(nb_repeat)+"_repeat/"+s+"_rt_reg.txt", reg)
	np.savetxt("../data/data_txt_"+str(nb_repeat)+"_repeat/"+s+".txt", new_data[s], fmt='%i')