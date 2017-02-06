#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import numpy as np
import sys
from time import sleep

def SoftMaxValues(values, beta):
    tmp = np.exp(values*float(beta))
    return  tmp/float(np.sum(tmp))


class qlearning_1():
    """ 
    """
    def __init__(self):
        self.n_action = 4
        self.n_r = 2
        self.max_entropy = -np.log2(1./self.n_action)
        self.bounds = dict({'alpha':[0.0, 1.0],
                    "beta":[0.0, 100.0], # QLEARNING
                    "sigma":[0.0, 20.0], 
                    "kappa":[0.0, 1.0],
                    "shift":[0.0, 1.0]})

    def sferes_call(self, sari, mean_rt, parameters):        
        self.parameters = parameters
        self.sari = sari[:,[3,4,5,9,2]] # reward | problem | action | index | phase
        self.N = len(self.sari)        
        self.mean_rt = mean_rt
        self.value = np.zeros(self.N)
        self.reaction = np.zeros(self.N)
        self.values_mf =  np.zeros(self.n_action)
        self.problem = self.sari[0,1]
        self.p_a_final = np.zeros(self.n_action)
        self.spatial_biases = np.ones(self.n_action) * (1./self.n_action)        
        for i in xrange(self.N):
            if self.sari[i][4]-self.sari[i-1][4] < 0.0 and i > 0:
                    # START BLOC
                    self.problem = self.sari[i][1]                    
                    # RESET Q-LEARNING SPATIAL BIASES AND REWARD SHIFT
                    # print "biais", self.spatial_biases
                    self.values_mf = self.spatial_biases/self.spatial_biases.sum()
                    self.values_mf[self.current_action] *= (1.0-self.parameters['shift'])
                    self.spatial_biases[self.sari[i,2]-1] += 1.0

            # START TRIAL
            self.current_action = self.sari[i][2]-1
            # print "PROBLEM=", self.problem, " ACTION=", self.current_action
            r = self.sari[i][0]            
            
            self.p_a_mf = np.exp(self.values_mf*float(self.parameters['beta']))            
            self.Hf = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
                    
            #nombre de inf
            ninf = np.isinf(self.p_a_mf).sum()  
            if np.isinf(self.p_a_mf).sum():
                self.p_a_mf = np.isinf(tmp)*((1.0/float(ninf))-ninf*0.0000001-0.0000001/ninf) + 0.0000001
            else :
                self.p_a_mf = self.p_a_mf/np.sum(self.p_a_mf)   
            
            if np.sum(self.p_a_mf == 0.0):
                self.p_a_mf+=1e-8;
                self.p_a_mf = self.p_a_mf/self.p_a_mf.sum()            


            self.value[i] = float(np.log(self.p_a_mf[self.current_action])) 
            # print self.value[i]
            H = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
            self.reaction[i] = H
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

    def test_call(self, list_of_problems, parameters):        
        self.parameters = parameters
        self.list_of_problems = list_of_problems
        self.N = len(self.list_of_problems)        

        self.value = np.zeros(self.N)
        self.reaction = np.zeros(self.N)
        self.values_mf =  np.zeros(self.n_action)
        self.p_a_final = np.zeros(self.n_action)        
        for i in xrange(self.N):        	
            # START BLOC
            self.problem = self.list_of_problems[i]            

            # SEARCH PHASE
            r = 0
            while r == 0:
            	self.p_a_mf = SoftMaxValues(self.values_mf, float(self.parameters['beta']))
            	self.current_action = self.sample(self.p_a_mf)
            	self.Hf = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
            	if self.current_action == self.problem:
            		r = 1
            	print "search", self.problem, self.current_action, r
            	self.updateValue(r)
            # REPEAT PHASE
            for j in xrange(3):
            	r = 0
            	self.p_a_mf = SoftMaxValues(self.values_mf, float(self.parameters['beta']))
            	self.current_action = self.sample(self.p_a_mf)
            	self.Hf = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
            	if self.current_action == self.problem:
            		r = 1
            	print "search", self.problem, self.current_action, r
            	self.updateValue(r)            	
            print "\n"
            
            
            
            



    def sample(self, values):
        tmp = [np.sum(values[0:i]) for i in range(len(values))]
        return np.sum(np.array(tmp) < np.random.rand())-1

    def updateValue(self, reward):
        r = int((reward==1)*1)
        # Updating model free
        r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0                        
        # print "R = ", float(r)
        # print self.values_mf[self.current_action]        
        self.delta = float(r)-self.values_mf[self.current_action]        
        # print "delta = ", self.delta
        self.values_mf[self.current_action] = self.values_mf[self.current_action]+self.parameters['alpha']*self.delta        
        # print " mf2=" , self.values_mf
        index = range(self.n_action)
        index.pop(int(self.current_action))        
        self.values_mf[index] = self.values_mf[index] + (1.0-self.parameters['kappa']) * (0.0 - self.values_mf[index])
        # print " mf3=", self.values_mf
