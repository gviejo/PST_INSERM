#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import numpy as np
import sys
from time import sleep

def SoftMaxValues(values, beta):
    tmp0 = values - np.max(values)
    tmp = np.exp(tmp0*float(beta))
    return  tmp/float(np.sum(tmp))

class bayesian_1():
    """ 
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


    def sferes_call(self, sari, mean_rt, parameters):        
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
        # FIT
        fit = np.zeros(2)
        fit[0] = np.sum(self.value)
        fit[1] = -np.sum(np.power(self.rt_model-self.mean_rt[:,1], 2.0))
        return fit

    def test_call(self, nb_repeat, list_of_problems, parameters):        
        self.parameters = parameters
        self.list_of_problems = list_of_problems[list_of_problems[:,0] == 1,1]
        self.N = len(self.list_of_problems)
        self.performance = np.zeros((nb_repeat, len(self.list_of_problems), 3))
        self.timing = np.zeros((nb_repeat, len(self.list_of_problems), 8))
        self.length = np.zeros((nb_repeat, len(self.list_of_problems)))

        for k in xrange(nb_repeat):
            self.value = np.zeros(self.N)
            self.reaction = np.zeros(self.N)
            self.values_mf =  np.zeros(self.n_action)
            self.p_a = np.zeros((int(self.parameters['length']), self.n_action))
            self.p_r_a = np.zeros((int(self.parameters['length']), self.n_action, 2))
            self.nb_inferences = 0
            self.n_element = 0
            self.Hb = self.max_entropy        
            self.uniform = np.ones((self.n_action, 2))*(1./(self.n_action*2))        
            self.p_a_final = np.zeros(self.n_action)

            for i in xrange(self.N):            
                # START BLOC
                self.problem = self.list_of_problems[i]
                self.n_element = 0
                r = 0

                # SEARCH PHASE
                count = 0
                while r == 0 and count < 5:
                    # BAYESIAN CALL
                    self.p = self.uniform[:,:]
                    self.Hb = self.max_entropy
                    self.nb_inferences = 0  
                    self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)                

                    while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:            
                        self.inferenceModule()
                        self.evaluationModule()                    
                    
                    self.current_action = int(self.sample(self.p_a_mb))
                    r = 1 if self.current_action == self.problem else 0                    
                    self.timing[k,i,count] = float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma'])+self.Hb)
                    
                    self.updateValue(r)
                    count += 1
                if r == 1:
                    self.length[k,i] = count - 1
                    # REPEAT PHASE
                    for j in xrange(3):
                        r = 0
                        # BAYESIAN CALL
                        self.p = self.uniform[:,:]
                        self.Hb = self.max_entropy
                        self.nb_inferences = 0  
                        self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)        
                        
                        while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:            
                            self.inferenceModule()
                            self.evaluationModule()                    
                        
                        self.current_action = self.sample(self.p_a_mb)
                        r = 1 if self.current_action == self.problem else 0                                             
                        self.updateValue(r)
                        self.performance[k,i,j] = r
                        self.timing[k,i,count] = float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma'])+self.Hb)                        

                        count += 1
                else:
                    self.length[k,i] = -1
                    
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
