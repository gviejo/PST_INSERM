#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pylab import *
import cPickle as pickle

monkeys =   { 'g':12701,
              'm':34752,
              'p':27692,
              'r':11634,
              's':13348 }

data = {}
for s in monkeys.keys():
	with open("../data/data_pickle/"+s+".pickle", 'rb') as f:
		data[s] = pickle.load(f)


# PLOT RT INC1 to N, puis COR1 to N

# SEARCH MAX N for each monkeys

rt = {}

for s in monkeys.keys():
# for s in ['m', 'p']:
	rt[s] = dict()
	search = [[data[s][0,6]]]	
	repeat = [[]]
	max_N_search = 0
	max_N_repeat = 0

	problem = data[s][0,4]
	for t in xrange(1, len(data[s])):
		if data[s][t,4] == problem:
			if data[s][t,2] == 0.0:
				search[-1].append(data[s][t,6])
			elif data[s][t,2] == 1.0:
				repeat[-1].append(data[s][t,6])
		else:			
			if len(search[-1]) > max_N_search:
				max_N_search = len(search[-1])
			if len(repeat[-1]) > max_N_repeat:
				max_N_repeat = len(repeat[-1])
			problem = data[s][t,4]
			search.append([data[s][t,6]])
			repeat.append([])


	for p in xrange(len(search)):
		for i in xrange(max_N_search-len(search[p])):
			search[p].append(0.0)
		for i in xrange(max_N_repeat-len(repeat[p])):\
			repeat[p].append(0.0)

	search = np.array(search)
	repeat = np.array(repeat)

	rt[s]['search'] = np.zeros((2,search.shape[1]))
	rt[s]['repeat'] = np.zeros((2,repeat.shape[1]))
	for i in xrange(search.shape[1]):
		rt[s]['search'][0,i] = np.mean(search[:,i][search[:,i] != 0.0])
		rt[s]['search'][1,i] = np.std(search[:,i][search[:,i] != 0.0])
	for i in xrange(repeat.shape[1]):
		rt[s]['repeat'][0,i] = np.mean(repeat[:,i][repeat[:,i] != 0.0])
		rt[s]['repeat'][1,i] = np.std(repeat[:,i][repeat[:,i] != 0.0])


figure()
cnt = 1
for s in rt.keys():
	subplot(5,2,cnt)	
	errorbar(range(len(rt[s]['search'][0])), rt[s]['search'][0], rt[s]['search'][1], label = s)
	# errorbar(range(5), rt[s]['search'][0,0:5], rt[s]['search'][1,0:5], label = s)
	subplot(5,2,cnt+1)	
	errorbar(range(len(rt[s]['repeat'][0])), rt[s]['repeat'][0], rt[s]['repeat'][1], label = s)
	# errorbar(range(5), rt[s]['repeat'][0,0:5], rt[s]['repeat'][1,0:5], label = s)
	legend()
	cnt += 2
show()