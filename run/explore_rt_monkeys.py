#!/usr/bin/env python
# -*- coding: utf-8 -*-

# EXPLORE THE BEST characteristics of rt

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
			# if len(search[-1]) > max_N_search: 
			# 	max_N_search = len(search[-1])
			# if len(repeat[-1]) > max_N_repeat:
			# 	max_N_repeat = len(repeat[-1])
			nb_incorrect = len(search) # number of search trial before correct
			if nb_incorrect in rt[s]['search'].keys(): # append search trial in corresponding error length
				rt[s]['search'][nb_incorrect].append(np.array(search))
				rt[s]['repeat'][nb_incorrect].append(np.array(repeat))
			else:
				rt[s]['search'][nb_incorrect] = [np.array(search)]
				rt[s]['repeat'][nb_incorrect] = [np.array(repeat)]
				
			problem = data[s][t,4]  # number of new problem
			search = [data[s][t,7]] 
			repeat = [] 

	for t in ['search']:
		for l in rt[s][t].keys():
			rt[s][t][l] = np.array(rt[s][t][l])
			rtmean[s][t][l] = np.zeros((2,l))
			rtmean[s][t][l][0] = rt[s][t][l].mean(0)
			rtmean[s][t][l][1] = rt[s][t][l].std(0)

	t = 'repeat'
	for l in rt[s][t].keys():
		meann = np.zeros((2,7))
		for i in xrange(7):
			tmp = []
			for j in xrange(len(rt[s][t][l])):
				if len(rt[s][t][l][j]) > i:
					tmp.append(rt[s][t][l][j][i])
			meann[0,i] = np.mean(tmp)
			meann[1,i] = sem(tmp)
				
		rtmean[s][t][l] = meann




figure(figsize = (20,15))

colors = {'p':'blue','s':'cyan','r':'black','m':'red','g':'green'}

for i in xrange(1,7):
	subplot(2,3,i)
	for s in rtmean.keys():
	# for s in ['p','s','r','m']:
	# for s in ['p','m']:
		# errorbar(range(i), rtmean[s]['search'][i][0], rtmean[s]['search'][i][1], fmt='o', label = str(s), color = colors[s])
		# errorbar(range(i,i+8), rtmean[s]['repeat'][i][0], rtmean[s]['repeat'][i][1], fmt='*', color = colors[s])
		plot(range(i), rtmean[s]['search'][i][0], 'o', label = str(s), color = colors[s])
		plot(range(i,i+7), rtmean[s]['repeat'][i][0], '*', color = colors[s])
		plot(range(i+7), np.hstack((rtmean[s]['search'][i][0],rtmean[s]['repeat'][i][0])), '-', color = colors[s])
	grid()
	title(str(i)+" SEARCH | 8 REPEAT")
	legend()

# for k in xrange(1,6):
# 	subplot(5,2,sub)
# 	for m in rtmean.keys():
# 		# errorbar(range(k), rtmean[m]['search'][k][0], rtmean[m]['search'][k][1], label = str(k))
# 		plot(range(k), rtmean[m]['search'][k][0], 'o-', label = str(m))
# 	legend()
# 	subplot(5,2,sub+1)
# 	for m in rtmean.keys():
# 		# errorbar(range(k), rtmean[m]['search'][k][0], rtmean[m]['search'][k][1], label = str(k))
# 		plot(range(8), rtmean[m]['repeat'][k][0,0:8], 'o-', label = str(m))	
# 	legend()

# 	sub+=2
show()