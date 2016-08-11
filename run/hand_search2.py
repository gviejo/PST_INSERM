#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To hand search a solution for m and p monkeys
# use of hebbian model for model-free

import sys
import os
import numpy as np
from pylab import *
sys.path.append("../src")

from Models import HFSelection 

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


model = HFSelection()

parameters = {	
	'alpha':0.01,

	'beta':2.0,
	'gamma':5.0,

	'noise':0.1,
	'length':5,

	'gain':1.1,
	'threshold':1.0, 
	
	'sigma':1.0, 
	}


model.analysis_call(monkeys['m'][0:2000], rt_reg_monkeys['m'][0:2000], parameters)


# INIT
model.n_element = 0


action_chain = [1,2,3,3,3,3]
reward_chain = [0,0,1,1,1,1]

max_entropy = [5,5,5,0,1,2]

rt = np.zeros(6)

various = np.zeros((6,10))
mat = np.zeros((3,6,parameters['length']+1))

for i in xrange(6):
	model.current_action = action_chain[i]
	r = reward_chain[i]
	# SEARCH OR REPEAT
	model.state = model.sari[i][4]
	model.p_a_hebbian = SoftMaxValues(model.values_hebbian[model.last_action+model.n_action*model.state], model.parameters['gamma'])    
	model.Hf = -(model.p_a_hebbian*np.log2(model.p_a_hebbian)).sum()
	# BAYESIAN CALL
	model.p = model.uniform[:,:]
	model.Hb = model.max_entropy
	model.nb_inferences = 0
	model.p_a_mb = np.ones(model.n_action)*(1./model.n_action)        
	model.p_decision = np.zeros(int(model.parameters['length'])+1)
	model.p_retrieval= np.zeros(int(model.parameters['length'])+1)
	model.p_sigmoide = np.zeros(int(model.parameters['length'])+1)
	model.p_ak = np.zeros(int(model.parameters['length'])+1)        
	
	reaction = np.zeros(int(model.parameters['length'])+1)
	# START            
	model.sigmoideModule()
	model.p_sigmoide[0] = model.pA
	model.p_decision[0] = model.pA
	model.p_retrieval[0] = 1.0-model.pA	

	model.fusionModule()	
	
	H = -(model.p_a_final*np.log2(model.p_a_final)).sum()    
	# reaction[0] = np.log2(0.25)+model.parameters['sigma']*model.Hf
	# reaction[0] = np.log2(model.nb_inferences+1.0) + H
	reaction[0] = model.nb_inferences
			
	for j in xrange(model.n_element):            		
		model.inferenceModule()
		model.evaluationModule()				
		model.fusionModule()                	
		# reaction[j+1] = model.Hb + model.parameters['sigma']*model.Hf        
		# reaction[j+1] = np.log2(model.nb_inferences+1.0)
		reaction[j+1] = model.nb_inferences
		model.sigmoideModule()
		model.p_sigmoide[j+1] = model.pA            
		model.p_decision[j+1] = model.pA*model.p_retrieval[j]            
		model.p_retrieval[j+1] = (1.0-model.pA)*model.p_retrieval[j]                            
		
		

	model.updateValue(reward_chain[i])	
	rt[i] = float(np.sum(reaction*np.round(model.p_decision.flatten(),3)))
	various[i,0] = model.Hf
	various[i,1] = model.Hb	
	
	mat[0,i] = model.p_decision
	mat[1,i] = model.p_retrieval
	mat[2,i] = model.p_sigmoide	




# FIGURE

figure()
subplot(331)
plot(rt_reg_monkeys['m'][9:15,1], 'o-', label = 'rt singe m')
legend()

subplot(334)
plot(rt, 'o-', label = 'rt model')	
legend()

subplot(337)
plot(various[:,0], 'o-', label = 'Hf')
plot(various[:,1], 'o-', label = 'Hb')
ylim(0, 2.1)
legend()

subplot(332)
imshow(mat[0].transpose(), origin = 'lower', interpolation = 'nearest', vmin = 0, vmax = 1)
xlabel("N")
ylabel("p(decision)")

subplot(335)
imshow(mat[1].transpose(), origin = 'lower', interpolation = 'nearest', vmin = 0, vmax = 1)
xlabel("N")
ylabel("p(retrieval)")

subplot(3,3,8)
imshow(mat[2].transpose(), origin = 'lower', interpolation = 'nearest', vmin = 0, vmax = 1)
xlabel("N")
ylabel("p(sigmoide)")

subplot(1,3,3)
imshow(model.values_hebbian, interpolation = 'nearest', vmin = 0, vmax = 1)


show()