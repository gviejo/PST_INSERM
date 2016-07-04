#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To hand search a solution for m and p monkeys

import sys
import os
import numpy as np
from pylab import *
sys.path.append("../src")

from Models import FSelection 

def SoftMaxValues(values, beta):
    tmp = np.exp(values*float(beta))
    return  tmp/float(np.sum(tmp))

monkeys = {}
N = {}        
rt_reg_monkeys = {}
for s in os.listdir("../data/data_txt_3_repeat/"):
	if "rt_reg.txt" in s:
		rt_reg_monkeys[s.split("_")[0]] = np.genfromtxt("../data/data_txt_3_repeat/"+s)
	else :
		monkeys[s.split(".")[0]] = np.genfromtxt("../data/data_txt_3_repeat/"+s)
		N[s.split(".")[0]] = len(monkeys[s.split(".")[0]])               


model = FSelection()

parameters = {	
	'alpha':0.5,

	'beta':2.0,
	'gamma':1.0,

	'noise':0.1,
	'length':5,

	'gain':1.0,
	'threshold':1.0, 
	
	'sigma':0.1, 
	'kappa':0.1,
	'shift':0.1
}


model.analysis_call(monkeys['m'][0:2000], rt_reg_monkeys['m'][0:2000], parameters)

# INIT
model.n_element = 0
# RESET Q-LEARNING SPATIAL BIASES AND REWARD SHIFT
model.values_mf = model.spatial_biases/model.spatial_biases.sum()
# shift bias
tmp = model.values_mf[0] # On suppose que 0 est la derniere action
model.values_mf *= model.parameters['shift']/3.
model.values_mf[0] = tmp*(1.0-model.parameters['shift'])

action_chain = [1,2,3,3,3,3]
reward_chain = [0,0,1,1,1,1]

rt = np.zeros(6)

various = np.zeros((6,10))

for i in xrange(6):
	model.current_action = action_chain[i]
	r = reward_chain[i]
	model.p_a_mf = SoftMaxValues(model.values_mf, parameters['gamma'])
	model.Hf = -(model.p_a_mf*np.log2(model.p_a_mf)).sum()
	# BAYESIAN CALL
	model.p = model.uniform[:,:]
	model.Hb = model.max_entropy
	model.nb_inferences = 0
	model.p_a_mb = np.ones(model.n_action)*(1./model.n_action)        
	model.p_decision = np.zeros(int(model.parameters['length'])+1)
	model.p_retrieval= np.zeros(int(model.parameters['length'])+1)
	model.p_sigmoide = np.zeros(int(model.parameters['length'])+1)
	model.p_ak = np.zeros(int(model.parameters['length'])+1)        
	q_values = np.zeros((int(model.parameters['length'])+1, model.n_action))
	reaction = np.zeros(int(model.parameters['length'])+1)
	# START            
	model.sigmoideModule()
	model.p_sigmoide[0] = model.pA
	model.p_decision[0] = model.pA
	model.p_retrieval[0] = 1.0-model.pA
	q_values[0] = model.p_a_mb

	model.fusionModule()
	model.p_ak[0] = model.p_a_final[model.current_action]            
	H = -(model.p_a_final*np.log2(model.p_a_final)).sum()    
	reaction[0] = np.log2(0.25)+model.parameters['sigma']*model.Hf
	model.Hb_list[i,0] = model.Hb
	for j in xrange(model.n_element):            
		model.inferenceModule()
		model.evaluationModule()
		model.Hb_list[i,j+1] = model.Hb
		q_values[j+1] = model.p_a_mb
		model.fusionModule()                
		model.p_ak[j+1] = model.p_a_final[model.current_action]                
		H = -(model.p_a_final*np.log2(model.p_a_final)).sum()
		N = model.nb_inferences+1.0        
		reaction[j+1] = model.Hb + model.parameters['sigma']*model.Hf        
		model.sigmoideModule()
		model.p_sigmoide[j+1] = model.pA            
		model.p_decision[j+1] = model.pA*model.p_retrieval[j]            
		model.p_retrieval[j+1] = (1.0-model.pA)*model.p_retrieval[j]                            
	
	rt[i] = float(np.sum(reaction*np.round(model.p_decision.flatten(),3)))
	various[i,0] = model.Hf
	various[i,1] = np.dot(model.p_decision, model.Hb_list[i])	
	model.updateValue(reward_chain[i])



# FIGURE

figure()
subplot(311)
plot(rt_reg_monkeys['m'][9:15,1], 'o-')

subplot(312)
plot(rt, 'o-')

subplot(313)
plot(various[:,0],'o-', label = 'Hf')
plot(various[:,1],'o-', label = 'Hb')
legend()
show()