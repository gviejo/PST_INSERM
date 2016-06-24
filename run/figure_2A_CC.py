#!/usr/bin/env python
# -*- coding: utf-8 -*-

# PLOT For mehdi suggestions
# to do representative steps

import numpy as np
from pylab import *
import cPickle as pickle
from scipy.stats import sem
import sys, os

# LOAD monkeys data
monkeys = {}
N = {}        
for s in os.listdir("../data/data_txt_3_repeat/"):
    if "rt_reg.txt" in s:
        pass
    else :
        monkeys[s.split(".")[0]] = np.genfromtxt("../data/data_txt_3_repeat/"+s)
        N[s.split(".")[0]] = len(monkeys[s.split(".")[0]])               

shift = {}

for s in monkeys.keys():
	shift[s] = 0
	start = monkeys[s][0,5]
	Ntrials = 0
	for i in xrange(1,N[s]):	
		if monkeys[s][i,2]-monkeys[s][i-1,2] < 0:			
			Ntrials += 1.0
			if start != monkeys[s][i,5]:				
				shift[s] += 1							
			start = monkeys[s][i,5]

	shift[s] = (shift[s]/Ntrials)*100.0



N = 5
shiftlist = tuple(np.sort(shift.values()))
staylist = tuple(100.0 - np.sort(shift.values()))

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, shiftlist, width, color='r')
p2 = plt.bar(ind, staylist, width, color='y', bottom=shiftlist)

plt.ylabel('Shift/stay')

plt.xticks(ind + width/2., np.array(shift.keys())[np.argsort(shift.values())])
plt.yticks(np.arange(0, 100, 10))
plt.legend((p1[0], p2[0]), ('Shift', 'Stay'))

plt.show()


