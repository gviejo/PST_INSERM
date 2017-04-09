#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

load and write for last_set for manuscrit


"""

import sys
import os
from optparse import OptionParser
import numpy as np
import multiprocessing
import itertools
sys.path.append("../sferes/last_set_models/")
from 	 fusion_1 import    fusion_1
from 	mixture_1 import   mixture_1
from   bayesian_1 import  bayesian_1
from  qlearning_1 import qlearning_1
from 	 fusion_2 import    fusion_2
from 	mixture_2 import   mixture_2
from  qlearning_2 import qlearning_2
from 	 fusion_3 import    fusion_3
from 	mixture_3 import   mixture_3
from  qlearning_3 import qlearning_3
from 	 fusion_4 import    fusion_4
from 	mixture_4 import   mixture_4
from  qlearning_4 import qlearning_4
from 	 fusion_5 import    fusion_5
from 	mixture_5 import   mixture_5
from   bayesian_5 import  bayesian_5
from 	 fusion_6 import    fusion_6
from 	mixture_7 import   mixture_7
from 	 fusion_7 import    fusion_7

sys.path.append("../../src")
from Models import *
# from matplotlib import *
# from pylab import *
from Sferes import pareto
from itertools import *
from time import sleep
import cPickle as pickle
import copy 


# ------------------------------------
# FUNCTION FOR MULTIPROCESSING
# ------------------------------------
nworker = 28
def worker_test_star(a_b):
	return worker_test(*a_b)

def worker_test(w, pos_, s, p_to_test_):
	value_ = np.zeros((len(pos_),2))

	data_1 = np.genfromtxt("../../data/data_txt_3_repeat/"+s+".txt")
	data_2 = np.genfromtxt("../../data/data_txt_3_repeat/"+s+"_rt_reg.txt")

	for t in pos_:
		m, v = p_to_test_[t].keys()[0].split(".") # model, version
		p = p_to_test_[t][p_to_test_[t].keys()[0]]		
		
		model = copy.deepcopy(vmodels[m][int(v)])
		fit = model.sferes_call(data_1, data_2, p)
		model.test_call(2, problems_sar[s], p)

		model.timing = model.timing - model.rt_align[0]
		model.timing = model.timing / model.rt_align[1]
		performance = {}
		timing = {}
		for i in xrange(5):
			index = model.length == i
			if index.sum():
				performance[i] = np.mean(model.performance[index], 0)
				timing[i] = np.mean(model.timing[index][:,0:i+4], 0)
		
				value_[np.where(pos_ == t)[0][0], 0] += np.sum(np.power(performance_monkeys[s][i+1]-performance[i], 2))		
				value_[np.where(pos_ == t)[0][0], 1] += np.sum(np.power(time_monkeys[s][i+1]-timing[i], 2))
		
		print "worker ", w, "| line ", t, " | value ", value_[np.where(pos_ == t)[0][0], 0], " ", np.where(pos_ == t)[0][0]		
	return np.hstack((np.vstack(pos_), value_))
	
# -----------------------------------
# VARIOUS
# -----------------------------------

models = dict({"fusion":FSelection(),
					"qlearning":QLearning(),
					"bayesian":BayesianWorkingMemory(),
					# "selection":KSelection(),
					"mixture":CSelection(),
					"metaf":MetaFSelection(),
					"sweeping":Sweeping()})
vmodels = dict({'fusion'	:	{	1	:	fusion_1()		,
									2	:	fusion_2()		,
									3	:	fusion_3()		,
									4	:	fusion_4()		,
									5	:	fusion_5()		,
									6	:	fusion_6()		,
									7	:	fusion_7()		
								},
				'mixture'	:	{	1	:	mixture_1()		,
									2	: 	mixture_2()		,
									3	: 	mixture_3()		,
									4	: 	mixture_4()		,
									5	: 	mixture_5()		,
									7	: 	mixture_7()		
								},
				'bayesian'	:	{	1	:	bayesian_1()	,
									5	:	bayesian_5()
								},
				'qlearning'	:	{	1	:	qlearning_1()	,
									2	:	qlearning_2()	,
									3	:	qlearning_3()	,
									4	:	qlearning_4()	,
								}
			})
nparameters = {'fusion'	:	{		1	:	8.0		,
									2	:	9.0		,
									3	:	9.0		,
									4	:	10.0	,
									5	:	10.0	,
									6	:	10.0	,
									7	:	12.0		
								},
				'mixture'	:	{	1	:	7.0		,
									2	: 	8.0		,
									3	: 	8.0		,
									4	: 	9.0		,
									5	: 	9.0		,
									7	: 	11.0		
								},
				'bayesian'	:	{	1	:	4.0		,
									5	:	4.0
								},
				'qlearning'	:	{	1	:	3.0		,
									2	:	4.0		,
									3	:	4.0		,
									4	:	5.0		,
								}
			}
p_order = dict({'fusion':['alpha','beta', 'noise','length', 'gain', 'threshold', 'gamma', 'sigma', 'kappa', 'shift', 'xi', 'yi'], 
					'qlearning':['alpha','beta', 'sigma', 'kappa', 'shift'],
					'bayesian':['length','noise','threshold', 'sigma'],
					'mixture':['alpha', 'beta', 'noise', 'length', 'weight', 'threshold', 'sigma', 'kappa', 'shift', 'xi', 'yi']})
bounds = dict({'fusion':	{"beta":[0.0, 100.0], # temperature for final decision                            
							'alpha':[0.0, 1.0],
							"length":[1, 10],
							"threshold":[0.0, 20.0], # sigmoide parameter
							"noise":[0.0, 0.1],
							"gain":[0.00001, 10000.0], # sigmoide parameter 
							"sigma":[0.0, 20.0],
							"gamma":[0.0, 100.0], # temperature for entropy from qlearning soft-max
							"kappa":[0.0, 1.0],
							"shift":[0.0, 0.999999],
							"xi":[-20, 0.0],
							"yi":[0.0, 20.0]},
				'mixture':	{"length":[1, 10], 
							"threshold":[0.01, -np.log2(1./4.)], 
							"noise":[0.0, 0.1],                                                
							'alpha':[0.0, 1.0],
							"beta":[0.0, 100.0], # QLEARNING
							"sigma":[0.0, 20.0], 
							"weight":[0.0, 1.0], 
							"kappa":[0.0, 1.0],
							"shift":[0.0, 0.999999],
							"xi":[-20, 0.0],
							"yi":[0.0, 20.0]},
				'bayesian':	{"length":[1, 10], 
                    		"threshold":[0.01, -np.log2(1./4.)], 
                    		"noise":[0.0, 0.1],                                                
                    		"sigma":[0.0, 20.0]},
                'qlearning' : {'alpha':[0.0, 1.0],
                    		"beta":[0.0, 100.0], # QLEARNING
                    		"sigma":[0.0, 20.0], 
                    		"kappa":[0.0, 1.0],
                    		"shift":[0.0, 1.0]}
				})




front = pareto("SFERES_9") # dummy for rt
test = np.arange(5)*3.0
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
to_compare_value = {}
#------------------------------------
# MONKEYS PERFORMANCE AND REACTION TIMES to get problems_sar
#------------------------------------
problems_sar = {}
performance_monkeys = {}
time_monkeys = {}
for s in monkeys:
	problems_sar[s] = []
	monkey = np.genfromtxt("../../data/data_txt_3_repeat/"+s+".txt", dtype = 'float')
	monkey[:,6] = monkey[:,6] - np.median(monkey[:,6])
	monkey[:,6] = monkey[:,6] / (np.percentile(monkey[:,6], 75) - np.percentile(monkey[:,6], 25))
	tmp = [[1, monkey[0,4]-1,monkey[0,5]-1,monkey[0,3]]]
	tmp3 = [] # for choices
	tmp4 = [monkey[0,6]] # for rt
	count = 0
	performance_monkeys[s] = []	
	time_monkeys[s] = {i:[] for i in xrange(1,6)}
	length_problems_count = []
	for i in xrange(1,len(monkey)):				
		if monkey[i-1,1] != monkey[i,1] or int(monkey[i,2]) == 0 and int(monkey[i-1,2]) == 1: # new problem 			
			tmp = np.array(tmp)
			if tmp[:,3].sum()>=1 and i - count > 1 and len(tmp4) > len(tmp):
				problems_sar[s] += list(tmp)							
				performance_monkeys[s].append(tmp3)
				time_monkeys[s][len(tmp4[0:-3])].append(tmp4)
				length_problems_count.append(len(tmp))				
			tmp = [[1, monkey[i,4]-1,monkey[i,5]-1,monkey[i,3]]]						
			tmp4 = [monkey[i,6]]
			count = i
		else:
			if int(monkey[i,2]) == 0 and int(monkey[i-1,2]) == 0: # search phase				
				tmp.append([0, monkey[i,4]-1,monkey[i,5]-1,monkey[i,3]])
				tmp4.append(monkey[i,6])		
			elif int(monkey[i,2]) == 1 and int(monkey[i-1,2]) == 0:# repeat phase					
				tmp3 = monkey[i:i+3,3]
				tmp4+=list(monkey[i:i+3,6])						
	problems_sar[s] = np.array(problems_sar[s])
	performance_monkeys[s] = np.array(performance_monkeys[s])
	for k in time_monkeys[s].keys():
		time_monkeys[s][k] = np.array(time_monkeys[s][k])
		time_monkeys[s][k] = np.mean(time_monkeys[s][k], 0)	
	tmp = {}
	for i in np.unique(length_problems_count):
		index = length_problems_count == i
		tmp[i] = np.mean(performance_monkeys[s][index], 0)							
	performance_monkeys[s] = tmp	


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
	for p in [1,2,3,4,5,6,7]: # ensemble testé
		data[s][p] = dict()
		pareto[s][p] = dict()		
		for m in set_to_models[p]: # modele dans ensemble testé
			data[s][p][id_to_models[m]] = dict()
			pareto[s][p][id_to_models[m]] = dict()
			for r in xrange(n_run):						
				data[s][p][id_to_models[m]][r] = np.genfromtxt("last_set/set_"+str(p)+"_"+str(m)+"/sferes_"+id_to_models[m]+"_pst_inserm_"+s+"_"+str(r)+"_"+str(p)+".dat")
				order = p_order[id_to_models[m]]
				if p < 6:
					order = order[0:-2]
				scale = bounds[id_to_models[m]]
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
			pareto[s][p][id_to_models[m]][:,3] = 2*pareto[s][p][id_to_models[m]][:,3] - nparameters[id_to_models[m]][p]*np.log(front.N[s])

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
		for p in [1,2,3,4,5,6,7]:
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

	data_run = data[s][set_][m][run_]
	tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]
	p_test[s] = {'best_tche':{m+str(set_):dict(zip(p_order[m],tmp[4:]))}}
	to_compare_value[s] = {'tche':value}
	position[s+str(set_)+'_tche'] = best_ind[5:]

# ------------------------------------
# SELECTION BY TESTING PARAMETERS
# ------------------------------------
	# FOR ALL SET
	# must call model testing files as determined by pareto3 = [model | set | run | gen | num | fit1 | fit2]
	# take only half percent of solutions according to value of tche		
	pool = multiprocessing.Pool(processes = nworker)
	cut = np.sort(value)[int(len(value)/4.)]
	points = np.where(value<cut)[0]
	pos = np.array_split(points, nworker)
	# find parameter for all point first; index dict by points array
	p_to_test = {}
	for t in points:
		l = pareto3[s][t]
		m = id_to_models[int(l[0])]
		set_ = int(best_ind[1])
		run_ = int(best_ind[2])
		gen_ = int(best_ind[3])
		num_ = int(best_ind[4])		
		data_run = data[s][set_][m][run_]
		tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]
		p_to_test[t] = { m+"."+str(set_) :  dict(zip(p_order[m],tmp[4:])) }

	value4 = pool.map(worker_test_star, itertools.izip(range(nworker), pos, itertools.repeat(s), itertools.repeat(p_to_test)))
	value4 = np.vstack(np.array(value4))
	value4[:,1:] = (value4[:,1:] - np.min(value4[:,1:], 0))/(np.max(value4[:,1:], 0) - np.min(value4[:,1:], 0))
	value4 = np.vstack((value4[:,0], np.sum(value4[:,1:], 1))).transpose()
	ind_best_point = int(value4[np.argmin(value4[:,1]), 0])	
	# pareto 3 has points with negative value must search in the pareto3 index	
	tmp = pareto3[s][:,5:]
	x = np.arange(len(tmp))
	index = (tmp[:,0]>0)*(tmp[:,1]>0)
	ind_best_point = x[index][ind_best_point]
	best_ind = pareto3[s][ind_best_point]

	# from data dictionnary
	m = id_to_models[int(best_ind[0])]
	set_ = int(best_ind[1])
	run_ = int(best_ind[2])
	gen_ = int(best_ind[3])
	num_ = int(best_ind[4])
	data_run = data[s][set_][m][run_]
	tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]
	p_test[s]['best_test'] = {m+str(set_):dict(zip(p_order[m],tmp[4:]))}
	to_compare_value[s]['test'] = value4
	position[s+str(set_)+'_test'] = best_ind[5:]
	
	

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
	tmp = tmp[positif]
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
	to_compare_value[s+"1"] = {'tche':value}
# ------------------------------------
# SELECTION BY TESTING PARAMETERS SET 1 only the best 25 percent of solutions according to value of tcheby
# ------------------------------------
	# pareto4 = model | run | gen | ind |	
	pool = multiprocessing.Pool(processes = nworker)
	cut = np.sort(value)[int(len(value)/4.)]
	points = np.where(value<cut)[0]
	pos = np.array_split(points, nworker)
	# find parameter for all point first; index dict by points array
	p_to_test = {}
	for t in points:
		l = pareto4[s][t]
		m = id_to_models[int(l[0])]
		run_ = int(l[1])
		gen_ = int(l[2])
		num_ = int(l[3])
		data_run = data[s][1][m][run_]		
		tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]
		p_to_test[t] = { m+".1" :  dict(zip(p_order[m],tmp[4:])) }


	value2 = pool.map(worker_test_star, itertools.izip(range(nworker), pos, itertools.repeat(s), itertools.repeat(p_to_test)))	
	value2 = np.vstack(np.array(value2))
	value2[:,1:] = (value2[:,1:] - np.min(value2[:,1:], 0))/(np.max(value2[:,1:], 0) - np.min(value2[:,1:], 0))
	value2 = np.vstack((value2[:,0], np.sum(value2[:,1:], 1))).transpose()
	ind_best_point = int(value2[np.argmin(value2[:,1]), 0])
	# pareto 4 has points with negative value
	tmp = pareto4[s][:,4:]
	x = np.arange(len(tmp))
	index = (tmp[:,0]>0)*(tmp[:,1]>0)
	ind_best_point = x[index][ind_best_point]
	best_ind = pareto4[s][ind_best_point]
	m = id_to_models[int(best_ind[0])]
	run_ = int(best_ind[1])
	gen_ = int(best_ind[2])
	num_ = int(best_ind[3])
	data_run = data[s][1][m][run_]
	tmp = data_run[(data_run[:,0] == gen_)*(data_run[:,1] == num_)][0]
	p_test_v1[s]['best_test'] = dict({m:dict(zip(p_order[m],tmp[4:]))})                        	
	tmp = np.ones(len(pareto4[s]))
	tmp[value2[:,0].astype('int')] = value2[:,1]
	to_compare_value[s+"1"]['test'] = tmp
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
with open("../papier/to_compare_value.pickle", 'wb') as f:
	pickle.dump(to_compare_value, f)
with open("../papier/p_test_all_v.pickle", 'wb') as f:
	pickle.dump(p_test, f)


with open("../papier/pareto2.pickle", 'wb') as f:
	pickle.dump(pareto2, f)
with open("../papier/pareto3.pickle", 'wb') as f:
	pickle.dump(pareto3, f)
with open("../papier/position.pickle", 'wb') as f:
	pickle.dump(position, f)

# with open("p_test_last_set.pickle", 'wb') as f:
# 	pickle.dump(p_test, f)
# with open("p_test2_last_set.pickle", 'wb') as f:
# 	pickle.dump(p_test2, f)
