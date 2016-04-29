#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

load and and plot multi objective results from Sferes 2 optimisation 


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np

sys.path.append("../../src")

from Models import FSelection

from matplotlib import *
from pylab import *

from Sferes import pareto
from itertools import *
from time import sleep

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test \n If none is provided, all files are loaded", default=False)
parser.add_option("-o", "--output", action="store", help="The output file of best parameters to test", default=False)
(options, args) = parser.parse_args()
# -----------------------------------

# -----------------------------------
# LOADING DATA
# -----------------------------------
front = pareto(options.input)
front.constructParetoFrontier('log')

figure()
i = 1
for m in front.pareto.keys():
	subplot(2,2,i)
	for s in front.pareto[m].keys():
		plot(front.pareto[m][s][:,3], front.pareto[m][s][:,4], 'o-')
	i+=1
	title(m)
	ylim(-20, 0.0)

figure()
subjects = front.pareto['qlearning'].keys()
i = 1
for s in subjects:
	subplot(2,3,i)
	for m in front.pareto.keys():
		if s in front.pareto[m].keys():
			plot(front.pareto[m][s][:,3], front.pareto[m][s][:,4], 'o-', label = m)
	i+=1
	legend(loc= 'best')
	title(s)		
	ylim(-20, 0.0)

# show()

figure()
i = 1
for s in front.monkeys.keys():
	subplot(2,3,i)
	plot(front.rt_reg_monkeys[s][:,1], 'o-', color = 'black')
	for m in front.pareto.keys():
	# for m in ['mixture']:
		if s in front.pareto[m].keys():
			for t in xrange(front.pareto[m][s].shape[0]):
				parameters = dict({k:v for k,v in zip(front.p_order[m],front.pareto[m][s][t][5:])})				
				model = front.models[m]
				fit = model.sferes_call(front.monkeys[s], front.rt_reg_monkeys[s], parameters)
				print s, m, t
				print fit[0], fit[1]
				print front.pareto[m][s][t][3], front.pareto[m][s][t][4]
				plot(model.rt_model, 'o-', alpha = 0.5)
				
	i+=1

show()
sys.exit()

parameters = dict({k:v for k,v in zip(front.p_order['fusion'],front.pareto['fusion']['p'][1][5:])})
model = FSelection()
fit = model.sferes_call(front.monkeys['p'], front.rt_reg_monkeys['p'], parameters)



print fit
print front.pareto['fusion']['p'][1][3], front.pareto['fusion']['p'][1][4]



sys.exit()
# front.showBrute()
front.removeIndivDoublons()
front.constructMixedParetoFrontier()
front.rankDistance()
front.rankOWA()
front.rankTchebytchev()
front.retrieveRanking()
front.timeConversion()
# front.writeParameters("parameters_2criterion.txt")
front.classifySubject()
front.preview()
# data_single, p_test_single = front.rankIndividualStrategy()
# timing_single = front.timeConversion_singleStrategy(p_test_single, data_single)
# show()
# sys.exit()
# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/pareto_front.pickle") , 'wb') as handle:    
#     pickle.dump(front.pareto, handle)

# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/mixed_pareto_front.pickle"), 'wb') as handle:    
#     pickle.dump(front.mixed, handle)

# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/rank_all_operators.pickle"), 'wb') as handle:
# 	pickle.dump(front.zoom, handle)

# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/parameters.pickle"), 'wb') as handle:
# 	pickle.dump(front.p_test, handle)

with open("parameters_meg.pickle", 'wb') as f:
	pickle.dump(front.p_test, f)

with open("timing_meg.pickle", 'wb') as f:
	pickle.dump(front.timing, f)

# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/rank_single.pickle"), 'wb') as handle:
# 	pickle.dump(data_single, handle)

# with open("parameters_single.pickle", 'wb') as f:
# 	pickle.dump(p_test_single, f)

# with open("timing_single.pickle", 'wb') as f:
#     pickle.dump(timing_single, f)

# fit to choice extremum of the front
with open("extremum.pickle", 'wb') as f:
	pickle.dump(front.p_test_extremum, f)

# value of maximum BIC normalized 
# with open(os.path.expanduser("~/Dropbox/ISIR/GoHal/Draft/data/obj_choice.pickle"), 'wb') as f:
#   	pickle.dump(front.choice_only, f)











sys.exit()
# BIC only 
figure()
s_to_plot = []
x_pos = []
tmp = 0
for x in front.choice_only.iterkeys():	
	for s in front.choice_only[x]:
		x_pos.append(len(s_to_plot)+tmp)
		s_to_plot.append(s)
		for m in front.data.iterkeys():			
			obj = front.data[m][s][0][:,2]-2000.0
			# obj = -2*obj+float(len(front.p_order[m]))*np.log(front.N)
			# obj = 1.0-obj/(156*np.log(0.2))
			ind = np.ones(len(obj))*(len(s_to_plot)+tmp+0.1*float(front.choice_only.keys().index(m)))
			plot(ind, obj, 'o', color = front.colors_m[m], markersize = 10, alpha = 0.8)
	tmp+=1
# ylim(0.0, 1.0)
xticks(np.array(x_pos)+1, s_to_plot)
show()


# front evolution

show()

def plotting():
	for s in front.mixed.keys():
		figure()
		for m in front.pareto.keys():
			data = front.pareto[m][s]
			plot(data[:,3], data[:,4], 'o-', color = front.colors_m[m], label = m)
		title(s)
		legend()
		show()

def gen_plotting(m, s):
	figure()
	n = 0
	color=iter(cm.rainbow(np.linspace(0,1,len(np.unique(front.data[m][s][n][:,0])))))
	for g in np.unique(front.data[m][s][n][:,0]):
		c = next(color)
		ind = front.data[m][s][n][:,0] == g
		gen = front.data[m][s][n][:,2:4][ind] - [2000.0,500.0]
		plot(gen[:,0], gen[:,1], 'o', c = c)
	plot(front.pareto[m][s][:,3] + float(len(front.p_order[m]))*np.log(front.N), front.pareto[m][s][:,4], 'o-')
	show()




