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

# quick test of qlearning sferes call
def alignToMedian(a):        
    self.model.reaction = self.model.reaction - np.median(self.model.reaction)
    self.model.reaction = self.model.reaction / (np.percentile(self.model.reaction, 75)-np.percentile(self.model.reaction, 25))        
def softMax(values):
    tmp = np.exp(values*beta)    
    return tmp/float(np.sum(tmp))

s = 'g'

alpha = 0.508241
beta = 0.0657192*100.0
sigma = 0.16

Q = np.zeros((4))
log = np.zeros(monkeys[s])
rt = np.zeros(monkeys[s])
q_values = np.zeros((monkeys[s],4))
rtm = data[s][:,-1]
for t in xrange(monkeys[s]):
# for t in xrange(6):
	state = int(data[s][t,4]-1)
	action =int(data[s][t,5]-1)
	reward =int(data[s][t,3])
	p_a = softMax(Q)	
	rt[t] = -(p_a*np.log2(p_a)).sum()	
	log[t] = np.log(p_a[action])
	q_values[t] = Q
	if (reward == 1):
		r = 1.0
	elif (reward == 0):
		r = -1.0
	
	delta = r - Q[action]
	Q[action] += alpha*delta

	# print rt[t], rtm[t]


rt = rt - np.median(rt)
rt = rt / (np.percentile(rt, 75) - np.percentile(rt, 25))

# for t in xrange(6):
# 	print rt[t], rtm[t]

print log.sum()+100000, -np.sum(np.power((rt - rtm), 2))+100000

# figure()
# plot(rt)
# plot(rtm)


# figure()
# plot(q_values)

# show()