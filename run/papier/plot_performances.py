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
performance = {}
for s in monkeys.keys():
	with open("../../data/data_pickle/"+s+".pickle", 'rb') as f:
		data[s] = pickle.load(f)
	performance[s] = []
	problem = data[s][0,1]	
	for i in xrange(1,monkeys[s]):
		if data[s][i,1] == problem:
			if int(data[s][i,2]) == 1 and int(data[s][i-1,2]) == 0:				
				performance[s].append(data[s][i:i+3,3])
		else:
			problem = data[s][i,1]
	if len(performance[s][-1]) != 3:
		performance[s].pop(-1)

	performance[s] = np.array(performance[s])



figure()
for s,i in zip(monkeys.keys(),range(1,6)):
	subplot(2,3,i)
	tmp = np.cumsum(performance[s],0)
	x = np.vstack(np.arange(1,len(tmp)+1))
	tmp = tmp/x
	for j in xrange(3):
		plot(tmp[:,j], label = str(j))
	ylim(0.7,1)
	legend()
	title(s)

show()