#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import numpy as np 
from pylab import *
import scipy.io as sp
import cPickle as pickle

monkeys = os.listdir("data_mat/")

trials_per_monkey = dict()
data = dict()

for a in monkeys:
	trials_per_monkey[a] = os.listdir("data_mat/"+a+"/")
	data[a] = []
	for t in trials_per_monkey[a]:
		data[a].append(sp.loadmat("data_mat/"+a+"/"+t)['DATA'])
	data[a] = np.vstack(np.array(data[a]))



for a in data.keys():

	rt = data[a][:,-2]
	rt = rt - np.median(rt)
	rt = rt / (np.percentile(rt, 75) - np.percentile(rt, 25))

	data[a] = np.hstack((data[a], np.vstack(rt)))

	with open("data_pickle/"+a+".pickle", 'wb') as f:
		pickle.dump(data[a], f)

	np.savetxt("data_txt/"+a+".txt", data[a], fmt='%f'  )	
	data[a].astype('int16').tofile("data_bin/"+a+".bin")




