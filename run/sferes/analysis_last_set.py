#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

load and write for last_set for manuscrit


"""

import sys
import os
from optparse import OptionParser
import numpy as np

sys.path.append("../../src")

from Models import *

from matplotlib import *
from pylab import *

from Sferes import pareto
from itertools import *
from time import sleep
import cPickle as pickle

models = dict({"fusion":FSelection(),
					"qlearning":QLearning(),
					"bayesian":BayesianWorkingMemory(),
					# "selection":KSelection(),
					"mixture":CSelection(),
					"metaf":MetaFSelection(),
					"sweeping":Sweeping()})

p_order = dict({'fusion':['alpha','beta', 'noise','length', 'gain', 'threshold', 'gamma', 'sigma', 'kappa', 'shift'], 
					'qlearning':['alpha','beta', 'sigma', 'kappa', 'shift'],
					'bayesian':['length','noise','threshold', 'sigma'],
					'selection':['beta','eta','length','threshold','noise','sigma', 'sigma_rt'],
					'mixture':['alpha', 'beta', 'noise', 'length', 'weight', 'threshold', 'sigma', 'kappa', 'shift'],
					'metaf':['alpha','beta', 'noise','length', 'gain', 'threshold', 'gamma', 'sigma', 'kappa', 'shift', 'eta'],
					'sweeping':['alpha','beta', 'noise','length', 'gain', 'threshold', 'gamma', 'sigma', 'kappa', 'shift']}) 


front = pareto("SFERES_9") # dummy for rt

# -----------------------------------
# LOADING DATA
# -----------------------------------
monkeys = ['m', 'p', 'g', 'r', 's']
id_to_models = dict({	1:'fusion',
						2:'mixture',
						3:'bayesian',
						4:'qlearning'})
models_to_id = dict({	'fusion':1,
						'mixture':2,
						'bayesian':3,
						'qlearning':4})

set_to_models = dict({	1:[1,2,3,4],
						2:[1,2,4],
						3:[1,2,4],
						4:[1,2,4],
						5:[1,2,3],
						6:[1],
						7:[1,2]})
n_run = 3
data = {}
pareto = {}
pareto2 = {}
pareto3 = {}
pareto4 = {}
p_test = {}
p_test2 = {}
p_test_v1 = {}
tche = {}
indd = {}
position = {}
#------------------------------------
# best log/rt
#------------------------------------
best_log = dict()
worst_log = dict()
for s in monkeys:
	best_log[s] = np.log(0.25)
	worst_log[s] = front.N[s]*np.log(0.25)
	problem = front.monkeys[s][0,4]
	for t in xrange(len(front.monkeys[s])):
		if front.monkeys[s][t,4] != problem:
			if front.monkeys[s][t,2] - front.monkeys[s][t-1,2] < 0.0:
				best_log[s] += np.log(0.25)

# ------------------------------------
# LOAD DATA
# ------------------------------------
for s in monkeys: # singe
	data[s] = dict()
	pareto[s] = dict() # first pareto set
	pareto2[s] = dict() # second pareto set with the set dimension
	pareto3[s] = dict() # third pareto set with mixed models
	# for p in set_to_models.iterkeys(): # ensemble testé
	for p in [1,2,3,4,5]: # ensemble testé
		data[s][p] = dict()
		pareto[s][p] = dict()		
		for m in set_to_models[p]: # modele dans ensemble testé
			data[s][p][id_to_models[m]] = dict()
			pareto[s][p][id_to_models[m]] = dict()
			for r in xrange(n_run):						
				data[s][p][id_to_models[m]][r] = np.genfromtxt("last_set/set_"+str(p)+"_"+str(m)+"/sferes_"+id_to_models[m]+"_pst_inserm_"+s+"_"+str(r)+"_"+str(p)+".dat")
				order = p_order[id_to_models[m]]
				scale = models[id_to_models[m]].bounds
				for i in order:
					data[s][p][id_to_models[m]][r][:,order.index(i)+4] = scale[i][0]+data[s][p][id_to_models[m]][r][:,order.index(i)+4]*(scale[i][1]-scale[i][0])

			part = data[s][p][id_to_models[m]]
			tmp={n:part[n][part[n][:,0]==np.max(part[n][:,0])] for n in part.iterkeys()}			
			tmp=np.vstack([np.hstack((np.ones((len(tmp[n]),1))*n,tmp[n])) for n in tmp.iterkeys()])			
			ind = tmp[:,3] != 0
			tmp = tmp[ind]
			tmp = tmp[tmp[:,3].argsort()][::-1]
			pareto_frontier = [tmp[0]]
			for pair in tmp[1:]:
				if pair[4] >= pareto_frontier[-1][4]:
					pareto_frontier.append(pair)
# pareto = run | gen | num | fit1 | fit2				
			pareto[s][p][id_to_models[m]] = np.array(pareto_frontier)
			pareto[s][p][id_to_models[m]][:,3] = pareto[s][p][id_to_models[m]][:,3] - 50000.0
			pareto[s][p][id_to_models[m]][:,4] = pareto[s][p][id_to_models[m]][:,4] - 50000.0            
			# bic
			pareto[s][p][id_to_models[m]][:,3] = 2*pareto[s][p][id_to_models[m]][:,3] - float(len(p_order[id_to_models[m]]))*np.log(front.N[s])

			best_bic = 2*best_log[s] - float(len(p_order[id_to_models[m]]))*np.log(front.N[s])
			worst_bic = 2*worst_log[s] - float(len(p_order[id_to_models[m]]))*np.log(front.N[s])                    
			pareto[s][p][id_to_models[m]][:,3] = (pareto[s][p][id_to_models[m]][:,3]-worst_bic)/(best_bic-worst_bic)	

			# rt
			if s == 'p':
				pareto[s][p][id_to_models[m]][:,4] = 1.0 - ((-pareto[s][p][id_to_models[m]][:,4])/(8.0*np.power(2.0*front.rt_reg_monkeys[s][:,1], 2).sum()))
			else :
				pareto[s][p][id_to_models[m]][:,4] = 1.0 - ((-pareto[s][p][id_to_models[m]][:,4])/(2.0*np.power(2.0*front.rt_reg_monkeys[s][:,1], 2).sum()))

# --------------------------------------
# MIXED PARETO FRONTIER between sets
# --------------------------------------
# pareto2 =   set | run | gen | num | fit1 | fit2				
	for m in id_to_models.iterkeys():
		tmp = {}	
		# for p in set_to_models.iterkeys():
		for p in [1,2,3,4,5]:
			if m in set_to_models[p]:
				tmp[p] = pareto[s][p][id_to_models[m]]
		tmp=np.vstack([np.hstack((np.ones((len(tmp[p]),1))*p,tmp[p])) for p in tmp.iterkeys()])			
		ind = tmp[:,4] != 0
		tmp = tmp[ind]
		tmp = tmp[tmp[:,4].argsort()][::-1]
		pareto_frontier = [tmp[0]]		
		for pair in tmp[1:]:
			if pair[5] >= pareto_frontier[-1][5]:
				pareto_frontier.append(pair)		
		pareto2[s][id_to_models[m]] = np.array(pareto_frontier)


# -------------------------------------
# MIXED PARETO FRONTIER between models
# ------------------------------------
# pareto3 = model | set | run | gen | num | fit1 | fit2		
	tmp = []
	for m in pareto2[s].iterkeys():		
		tmp.append(np.hstack((np.ones((len(pareto2[s][m]),1))*models_to_id[m], pareto2[s][m][:,0:6])))            	
	tmp = np.vstack(tmp)
	tmp = tmp[tmp[:,5].argsort()][::-1]                        
	if len(tmp):
		pareto3[s] = []
		pareto3[s] = [tmp[0]]
		for pair in tmp[1:]:
			if pair[6] >= pareto3[s][-1][6]:
				pareto3[s].append(pair)
		pareto3[s] = np.array(pareto3[s])            	

# -------------------------------------
# TCHEBYTCHEV
# -------------------------------------	
	tmp = pareto3[s][:,5:]
	tmp = tmp[(tmp[:,0]>0)*(tmp[:,1]>0)]
	ideal = np.max(tmp, 0)
	nadir = np.min(tmp, 0)
	value = 0.5*((ideal-tmp)/(ideal-nadir))
	value = np.max(value, 1)+0.001*np.sum(value,1)
	tche[s] = value
	ind_best_point = np.argmin(value)
	# Saving best individual
	best_ind = pareto3[s][ind_best_point]
	indd[s] = best_ind	
	
	
	# from data dictionnary
	m = id_to_models[int(best_ind[0])]
	set_ = int(best_ind[1])
	run_ = int(best_ind[2])
	gen_ = int(best_ind[3])
	num_ = int(best_ind[4])

	# print s
	# print "set ", set_
	# print "run ", run_
	# print "gen ", gen_
	# print "num ", num_

	data_run = data[s][set_][m][run_]
	tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]
	p_test[s+str(set_)] = dict({m:dict(zip(p_order[m],tmp[4:]))})                        
	position[s+str(set_)] = best_ind[5:]

# ------------------------------------
# SELECTION BY TESTING PARAMETERS
# ------------------------------------
	# must call testing files

# ------------------------------------
# BEST RT
# ------------------------------------
	index = (pareto3[s][:,5] > 0)*(pareto3[s][:,6] > 0)
	tmp = pareto3[s][index,:]
	best_ind = tmp[-1]
	m = id_to_models[int(best_ind[0])]
	set_ = int(best_ind[1])
	run_ = int(best_ind[2])
	gen_ = int(best_ind[3])
	num_ = int(best_ind[4])

	print s
	print "set ", set_
	print "run ", run_
	print "gen ", gen_
	print "num ", num_
	print "model" , m

	data_run = data[s][set_][m][run_]
	tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]
	p_test2[s+str(set_)] = dict({m:dict(zip(p_order[m],tmp[4:]))})
	
# -------------------------------------
# PARETO SET 1
# -------------------------------------
# pareto4 = model | run | gen | ind |
	tmp = []
	for m in pareto[s][1].iterkeys():
		tmp.append(np.hstack((np.ones((len(pareto[s][1][m]),1))*models_to_id[m], pareto[s][1][m][:,0:5])))            	
	tmp = np.vstack(tmp)
	tmp = tmp[tmp[:,4].argsort()][::-1]
	if len(tmp):
		pareto4[s] = []
		pareto4[s] = [tmp[0]]
		for pair in tmp[1:]:
			if pair[5] >= pareto4[s][-1][5]:
				pareto4[s].append(pair)
		pareto4[s] = np.array(pareto4[s])

# ------------------------------------
# TCHEBYTCHEV SET 1
# ------------------------------------
	tmp = pareto4[s][:,4:]
	positif = (tmp[:,0]>0)*(tmp[:,1]>0)
	tpm = tmp[positif]
	ideal = np.max(tmp[:,0:2], 0)
	nadir = np.min(tmp[:,0:2], 0)
	value = 0.5*((ideal-tmp)/(ideal-nadir))
	value = np.max(value, 1)+0.001*np.sum(value,1)
	ind_best_point = np.argmin(value)
	# Saving best individual
	best_ind = pareto4[s][ind_best_point]
	m = id_to_models[int(best_ind[0])]
	run_ = int(best_ind[1])
	gen_ = int(best_ind[2])
	num_ = int(best_ind[3])
	data_run = data[s][1][m][run_]
	tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]
	p_test_v1[s] = {'best_tche':dict({m:dict(zip(p_order[m],tmp[4:]))})}                        

# ------------------------------------
# BEST CHOICE & RT SET 1
# ------------------------------------
	tmp = {}
	for m in models_to_id.keys():
		bic_ = pareto[s][1][m][0,3]*(best_bic-worst_bic)+worst_bic
		tmp[m+"_"+str(bic_)] = dict(zip(p_order[m],pareto[s][1][m][0,5:]))
	p_test_v1[s]['best_choice'] = tmp	
	

# # SAVING IN ../papier/	
with open("../papier/p_test_v1.pickle",'wb') as f:
	pickle.dump(p_test_v1, f)
with open("../papier/p_test_all_v.pickle", 'wb') as f:
	pickle.dump(p_test)

# # SAVING IN DROPBOX
# with open("/home/viejo/Dropbox/Manuscrit/Chapitre5/monkeys/pareto2.pickle", 'wb') as f:
# 	pickle.dump(pareto2, f)
# with open("/home/viejo/Dropbox/Manuscrit/Chapitre5/monkeys/pareto3.pickle", 'wb') as f:
# 	pickle.dump(pareto3, f)
# with open("/home/viejo/Dropbox/Manuscrit/Chapitre5/monkeys/position.pickle", 'wb') as f:
# 	pickle.dump(position, f)

# with open("p_test_last_set.pickle", 'wb') as f:
# 	pickle.dump(p_test, f)
# with open("p_test2_last_set.pickle", 'wb') as f:
# 	pickle.dump(p_test2, f)
