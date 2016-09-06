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

class BayesianWorkingMemory():
    """ Bayesian Working memory strategy
    """

    def __init__(self):
        self.n_action = 4
        self.n_r = 2
        self.max_entropy = -np.log2(1./self.n_action)
        self.bounds = dict({"length":[1, 10], 
                    "threshold":[0.01, self.max_entropy], 
                    "noise":[0.0, 0.1],                                                
                    "sigma":[0.0, 20.0]})

    def analysis_call(self, sari, mean_rt, parameters):        
        self.parameters = parameters
        self.sari = sari[:,[3,4,5,9,2]] # reward | problem | action | index | phase
        self.N = len(self.sari)        
        self.mean_rt = mean_rt
        self.value = np.zeros(self.N)
        self.reaction = np.zeros(self.N)
        self.p_a = np.zeros((int(self.parameters['length']), self.n_action))
        self.p_r_a = np.zeros((int(self.parameters['length']), self.n_action, 2))
        self.nb_inferences = 0
        self.n_element = 0
        self.Hb = self.max_entropy        
        self.uniform = np.ones((self.n_action, 2))*(1./(self.n_action*2))
        self.problem = self.sari[0,1]
        self.p_a_final = np.zeros(self.n_action)
        ## LIST ####        
        self.entropy_list = np.zeros((self.N,2))
        self.free_list = np.zeros((self.N,4))
        self.biais_list = np.zeros((self.N,4))
        self.delta_list = np.zeros((self.N,4))
        self.inference_list = np.zeros((self.N,1))
        ############
        for i in xrange(self.N):
            if self.sari[i][4]-self.sari[i-1][4] < 0.0 and i > 0:
                    # START BLOC
                    self.problem = self.sari[i][1]
                    self.n_element = 0

            # START TRIAL
            self.current_action = self.sari[i][2]-1
            # print i, "PROBLEM=", self.problem, " ACTION=", self.current_action
            r = self.sari[i][0]            
                        
            # BAYESIAN CALL
            self.p = self.uniform[:,:]
            self.Hb = self.max_entropy
            self.nb_inferences = 0  
            self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)        

            while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:            
                self.inferenceModule()
                self.evaluationModule()                    
                # print self.p_a_mb
                
            ##########################            
            self.entropy_list[i,0] = self.Hb            
            self.free_list[i] = self.p_a_mb            
            self.inference_list[i] = self.nb_inferences            
            ##########################

            self.value[i] = float(np.log(self.p_a_mb[self.current_action])) 
            # print self.value[i]
            H = -(self.p_a_mb*np.log2(self.p_a_mb)).sum()
            # self.reaction[i] = float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma'])+H)
            self.reaction[i] = self.parameters['sigma']*self.Hb
            # print self.reaction[i]
            self.updateValue(r)

        # ALIGN TO MEDIAN
        self.reaction = self.reaction - np.median(self.reaction)
        self.reaction = self.reaction / (np.percentile(self.reaction, 75)-np.percentile(self.reaction, 25))        
        # LEAST SQUARES            
        self.rt_model = np.zeros(len(self.mean_rt))
        for i in xrange(len(self.rt_model)):
            self.rt_model[i] = np.mean(self.reaction[self.sari[:,3] == i])

    def sample(self, values):
        tmp = [np.sum(values[0:i]) for i in range(len(values))]
        return np.sum(np.array(tmp) < np.random.rand())-1

    def inferenceModule(self):                
        tmp = self.p_r_a[self.nb_inferences] * np.vstack(self.p_a[self.nb_inferences])
        self.p = self.p + tmp
        self.nb_inferences+=1

    def evaluationModule(self):        
        p_ra = self.p/np.sum(self.p) 
        p_r = np.sum(p_ra, axis = 0)
        p_a_r = p_ra/p_r
        self.p_a_mb = p_a_r[:,1]/p_a_r[:,0]
        self.p_a_mb = self.p_a_mb/np.sum(self.p_a_mb)
        self.Hb = -np.sum(self.p_a_mb*np.log2(self.p_a_mb))

    def updateValue(self, reward):
        r = int((reward==1)*1)
        if self.parameters['noise']:
            self.p_a = self.p_a*(1-self.parameters['noise'])+self.parameters['noise']*(1.0/self.n_action*np.ones(self.p_a.shape))
            self.p_r_a = self.p_r_a*(1-self.parameters['noise'])+self.parameters['noise']*(0.5*np.ones(self.p_r_a.shape))
        #Shifting memory            
        if self.n_element < int(self.parameters['length']):
            self.n_element+=1
        self.p_a[1:self.n_element] = self.p_a[0:self.n_element-1]
        self.p_r_a[1:self.n_element] = self.p_r_a[0:self.n_element-1]
        self.p_a[0] = np.ones(self.n_action)*(1/float(self.n_action))
        self.p_r_a[0] = np.ones((self.n_action, 2))*0.5
        #Adding last choice                 
        self.p_a[0] = 0.0
        self.p_a[0, self.current_action] = 1.0
        self.p_r_a[0, self.current_action] = 0.0
        self.p_r_a[0, self.current_action, int(r)] = 1.0        






monkeys = {}
N = {}        
rt_reg_monkeys = {}
for s in os.listdir("../data/data_txt_3_repeat/"):
	if "rt_reg.txt" in s:
		rt_reg_monkeys[s.split("_")[0]] = np.genfromtxt("../data/data_txt_3_repeat/"+s)
	else :
		monkeys[s.split(".")[0]] = np.genfromtxt("../data/data_txt_3_repeat/"+s)
		N[s.split(".")[0]] = len(monkeys[s.split(".")[0]])               


model = BayesianWorkingMemory()
# model = FSelection()

parameters = {	'length':4, 
				'threshold': 1.8, 
				'noise':0.01,
				'sigma':0.01 }
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
# Hb_after = {}

various = np.zeros((6,10))
mat = np.zeros((3,6,parameters['length']+1))

for p in xrange(5): #pour chaque probleme
			
	model.p_a = np.zeros((int(model.parameters['length']), model.n_action))
	model.p_r_a = np.zeros((int(model.parameters['length']), model.n_action, 2))
	model.nb_inferences = 0
	model.n_element = 0
	model.Hb = model.max_entropy        
	model.uniform = np.ones((model.n_action, 2))*(1./(model.n_action*2))
	model.problem = model.sari[0,1]
	model.p_a_final = np.zeros(model.n_action)			
	model.n_element = 0	
	model.p = model.uniform[:,:]
	hb_list = np.zeros(len(action_chain[p]))
	N_list = np.zeros(len(action_chain[p]))
	reaction = np.zeros(len(action_chain[p]))
	for i in xrange(len(action_chain[p])):		

		model.current_action = action_chain[p][i]
		r = reward_chain[p][i]
		# BAYESIAN CALL		
		model.nb_inferences = 0
		if len(action_chain[p]) -i <= 3:
			print p, i, "no preswiping"
			model.Hb = model.max_entropy		
			model.p = model.uniform[:,:]
			model.p_a_mb = np.ones(model.n_action)*(1./model.n_action)        		
		else :
			print p, i, "preswepping ", model.Hb
		# START            				
		while model.Hb > model.parameters['threshold'] and model.nb_inferences < model.n_element:
			model.inferenceModule()
			model.evaluationModule()				
			
		hb_list[i] = model.Hb		
		N_list[i] = model.nb_inferences									
		reaction[i] = np.log(model.nb_inferences+1.0) + model.Hb*0.51
		N_list[i] = model.nb_inferences
		hb_list[i] = model.Hb

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
