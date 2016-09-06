#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Trying sweeping between trial

import sys
import os
import numpy as np
from pylab import *
sys.path.append("../src")

def SoftMaxValues(values, beta):
	tmp = np.exp(values*float(beta))
	return  tmp/float(np.sum(tmp))

from Models import FSelection


monkeys = {}
N = {}        
rt_reg_monkeys = {}
for s in os.listdir("../data/data_txt_3_repeat/"):
	if "rt_reg.txt" in s:
		rt_reg_monkeys[s.split("_")[0]] = np.genfromtxt("../data/data_txt_3_repeat/"+s)
	else :
		monkeys[s.split(".")[0]] = np.genfromtxt("../data/data_txt_3_repeat/"+s)
		N[s.split(".")[0]] = len(monkeys[s.split(".")[0]])               


# model = BayesianWorkingMemory()
model = FSelection()

parameters = {	'length':4, 
				'threshold': 1.8, 
				'noise':0.01,
				'sigma':0.01,

				'alpha':0.01,

				'beta':2.0,
				'gamma':5.0,
				
				'length':4,

				'gain':1.1,
				'threshold':1.0, 

				'kappa': 0.5,
				'shift': 0.1
				}


model.analysis_call(monkeys['p'][0:1000], rt_reg_monkeys['p'][0:1000], parameters)


action_chain = {0:[0,0,0,0],
				1:[0,1,1,1,1],
				2:[0,1,2,2,2,2],
				3:[0,1,2,3,3,3,3],
				4:[0,1,2,1,3,3,3,3]}
reward_chain = {0:[1,1,1,1],
				1:[0,1,1,1,1],
				2:[0,0,1,1,1,1],
				3:[0,0,0,1,1,1,1],
				4:[0,0,0,0,1,1,1,1]}
rt = {}
N = {}
Hb = {}
H = {}

various = np.zeros((6,10))
mat = np.zeros((3,6,parameters['length']+1))

for p in xrange(5): #pour chaque probleme
	# START BLOC
	model.n_element = 0		
	# RESET Q-LEARNING SPATIAL BIASES AND REWARD SHIFT
	model.values_mf = model.spatial_biases/model.spatial_biases.sum()
	# shift bias
	tmp = model.values_mf[model.current_action]
	model.values_mf *= model.parameters['shift']/3.
	model.values_mf[model.current_action] = tmp*(1.0-model.parameters['shift'])
	# spatial biaises update
	# model.spatial_biases[model.sari[i,2]-1] += 1.0
	

	# model.p_a = np.zeros((int(model.parameters['length']), model.n_action))
	# model.p_r_a = np.zeros((int(model.parameters['length']), model.n_action, 2))
	# model.nb_inferences = 0	

	hb_list = np.zeros(len(action_chain[p]))
	N_list = np.zeros(len(action_chain[p]))	
	reaction = np.zeros(len(action_chain[p]))

	for i in xrange(len(action_chain[p])):		

		model.current_action = action_chain[p][i]
		r = reward_chain[p][i]
		# QLEARNING CALL
		model.p_a_mf = SoftMaxValues(model.values_mf, model.parameters['gamma'])    
		model.Hf = -(model.p_a_mf*np.log2(model.p_a_mf)).sum()
		# BAYESIAN CALL		
		model.nb_inferences = 0
		reaction_values = np.zeros(int(model.parameters['length'])+1) 						
		model.p_decision = np.zeros(int(model.parameters['length'])+1)
		model.p_retrieval= np.zeros(int(model.parameters['length'])+1)
		model.p_sigmoide = np.zeros(int(model.parameters['length'])+1)
		model.p_ak = np.zeros(int(model.parameters['length'])+1)        
		q_values = np.zeros((int(model.parameters['length'])+1, model.n_action))		
		h_final = np.zeros(int(model.parameters['length'])+1)  
		if len(action_chain[p]) -i <= 3 or i == 0:
			# BAYESIAN UPDATE
			model.p = model.uniform[:,:]
			model.Hb = model.max_entropy
			model.p_a_mb = np.ones(model.n_action)*(1./model.n_action)        			
			print p, i, "no preswiping", model.Hb
		else :
			print p, i, "preswepping ", model.Hb
			# model.Hb et model.p_a_mb ont déja été calculés
			# du coup on appele une fusion par défaut
		# START            
		model.sigmoideModule()
		model.p_sigmoide[0] = model.pA
		model.p_decision[0] = model.pA
		model.p_retrieval[0] = 1.0-model.pA            
		model.fusionModule()
		model.p_ak[0] = model.p_a_final[model.current_action]            
		H = -(model.p_a_final*np.log2(model.p_a_final)).sum()
		print H
		reaction_values[0] = float(((np.log2(model.nb_inferences+1.0))**model.parameters['sigma'])+H)
		h_final[0] = model.Hb
		
		for j in xrange(model.n_element):
			model.inferenceModule()
			model.evaluationModule()
			model.fusionModule()
			model.p_ak[j+1] = model.p_a_final[model.current_action]                
			H = -(model.p_a_final*np.log2(model.p_a_final)).sum()			
			h_final[j] = H
			reaction_values[j+1] = float(((np.log2(model.nb_inferences+1.0))**model.parameters['sigma'])+H)
			# reaction[j+1] = H
			model.sigmoideModule()
			model.p_sigmoide[j+1] = model.pA            
			model.p_decision[j+1] = model.pA*model.p_retrieval[j]            
			model.p_retrieval[j+1] = (1.0-model.pA)*model.p_retrieval[j]                    
				
		hb_list[i] = np.dot(h_final, model.p_decision)
		N_list[i] = np.dot(np.arange(int(model.parameters['length'])+1), model.p_decision)									
		reaction[i] = np.dot(reaction_values, model.p_decision)

		# UPDATE
		model.updateValue(r)	

		# sweeping
		if len(action_chain[p]) - i > 3:						
			model.Hb = model.max_entropy
			model.p = model.uniform[:,:]
			model.nb_inferences = 0
			while model.nb_inferences < model.n_element:
				model.inferenceModule()
			model.evaluationModule()		
		
	reaction = np.array(reaction)*1.0
	reaction = reaction - np.median(reaction)
	reaction = reaction / (np.percentile(reaction, 75) - np.percentile(reaction, 25))

	rt[p] = np.array(reaction)
	N[p] = np.array(N_list)
	Hb[p] = np.array(hb_list)



# FIGURE
figure(figsize = (16,5))
for p in xrange(5):
	subplot(2,5,p+1)
	plot(rt_reg_monkeys['p'][rt_reg_monkeys['p'][:,0] == p+1,1], 'o-', color = 'black')
	plot(rt[p], 'o-')
	axvline(p+0.5)
	subplot(2,5,p+1+5)
	plot(N[p], label = "N")
	plot(Hb[p], label = "Hb")
	axvline(p+0.5)
	legend()


show()
