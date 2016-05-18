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


class FSelection():
    """ fusion strategy

    """
    def __init__(self):
        #Parameters
        self.n_action = 4
        self.n_r = 2
        self.bounds = dict({"beta":[0.0, 100.0], # temperature for final decision                            
                            'alpha':[0.0, 1.0],
                            "length":[1, 10],
                            "threshold":[0.0, 20.0], # sigmoide parameter
                            "noise":[0.0, 0.1],
                            "gain":[0.00001, 10000.0], # sigmoide parameter 
                            "sigma":[0.0, 20.0],
                            "gamma":[0.0, 100.0], # temperature for entropy from qlearning soft-max
                            "kappa":[0.0, 1.0],
                            "shift":[0.0, 1.0]})

    def sferes_call(self, sari, mean_rt, parameters):
        self.parameters = parameters
        self.sari = sari[:,[3,4,5,9,2]] # reward | problem | action | index | phase
        self.N = len(self.sari)
        self.max_entropy = -np.log2(1./self.n_action)        
        self.mean_rt = mean_rt
        self.value = np.zeros(self.N)
        self.reaction = np.zeros(self.N)        
        self.values_mf = np.zeros(self.n_action)
        self.p_a = np.zeros((int(self.parameters['length']), self.n_action))
        self.p_r_a = np.zeros((int(self.parameters['length']), self.n_action, 2))
        self.nb_inferences = 0
        self.n_element = 0
        self.Hb = self.max_entropy
        self.Hf = self.max_entropy    
        self.uniform = np.ones((self.n_action, 2))*(1./(self.n_action*2))
        self.problem = self.sari[0,1]
        self.p_a_final = np.zeros(self.n_action)
        self.spatial_biases = np.ones(self.n_action) * (1./self.n_action)        
        for i in xrange(self.N):        
            if self.sari[i][1] != self.problem:                
                if self.sari[i][4]-self.sari[i-1][4] < 0.0:                    
                    # START BLOC
                    self.problem = self.sari[i][1]
                    self.n_element = 0
                    # RESET Q-LEARNING SPATIAL BIASES AND REWARD SHIFT
                    self.values_mf = self.spatial_biases/self.spatial_biases.sum()
                    # shift bias
                    tmp = self.values_mf[self.current_action]
                    self.values_mf *= self.parameters['shift']/3.
                    self.values_mf[self.current_action] = tmp*(1.0-self.parameters['shift'])
                    # spatial biaises update
                    self.spatial_biases[self.sari[i,2]-1] += 1.0


            # START TRIAL
            self.current_action = self.sari[i][2]-1
            r = self.sari[i][0]            

            self.p_a_mf = SoftMaxValues(self.values_mf, self.parameters['gamma'])    
            self.Hf = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
            # BAYESIAN CALL
            self.p = self.uniform[:,:]
            self.Hb = self.max_entropy
            self.nb_inferences = 0
            self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)        
            self.p_decision = np.zeros(int(self.parameters['length'])+1)
            self.p_retrieval= np.zeros(int(self.parameters['length'])+1)
            self.p_sigmoide = np.zeros(int(self.parameters['length'])+1)
            self.p_ak = np.zeros(int(self.parameters['length'])+1)        
            q_values = np.zeros((int(self.parameters['length'])+1, self.n_action))
            reaction = np.zeros(int(self.parameters['length'])+1)
            # START            
            self.sigmoideModule()
            self.p_sigmoide[0] = self.pA
            self.p_decision[0] = self.pA
            self.p_retrieval[0] = 1.0-self.pA
            # print "mf = ", self.values_mf

            self.fusionModule()
            self.p_ak[0] = self.p_a_final[self.current_action]            
            H = -(self.p_a_final*np.log2(self.p_a_final)).sum()
            reaction[0] = float(H)        
            for j in xrange(self.n_element):            
                self.inferenceModule()
                self.evaluationModule()
                # print "mf = ", self.values_mf
                self.fusionModule()                
                self.p_ak[j+1] = self.p_a_final[self.current_action]                
                H = -(self.p_a_final*np.log2(self.p_a_final)).sum()
                N = self.nb_inferences+1.0
                reaction[j+1] = float(((np.log2(N))**self.parameters['sigma'])+H)
                # reaction[j+1] = H
                self.sigmoideModule()
                self.p_sigmoide[j+1] = self.pA            
                self.p_decision[j+1] = self.pA*self.p_retrieval[j]            
                self.p_retrieval[j+1] = (1.0-self.pA)*self.p_retrieval[j]                    
                # print j+1, " p_ak=", self.p_ak[j+1], " p_decision=", self.p_decision[j+1], " p_retrieval=", self.p_retrieval[0]
            
            # print np.dot(self.p_decision,self.p_ak)
            self.value[i] = float(np.log(np.dot(self.p_decision,self.p_ak)))        
            # print self.value[i]
            # print self.p_decision
            self.reaction[i] = float(np.sum(reaction*np.round(self.p_decision.flatten(),3)))
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
        p_a_mb = self.p_a_mb/np.sum(self.p_a_mb)
        self.Hb = -np.sum(p_a_mb*np.log2(p_a_mb))

    def sigmoideModule(self):
        np.seterr(invalid='ignore')
        x = 2*self.max_entropy-self.Hb-self.Hf
        self.pA = 1/(1+((self.n_element-self.nb_inferences)**self.parameters['threshold'])*np.exp(-x*self.parameters['gain']))
        # print "n=",self.n_element," i=", self.nb_inferences, " Hb=", self.Hb, " Hf=", self.Hf, " x=", x, " p(A)=",self.pA, "threshold= ", self.parameters['threshold'], "gain = ", self.parameters['gain']
        return np.random.uniform(0,1) > self.pA
    
    def fusionModule(self):
        np.seterr(invalid='ignore')
        self.values_net = self.p_a_mb+self.values_mf
        tmp = np.exp(self.values_net*float(self.parameters['beta']))
        
        # print "fusion ", self.values_mf[self.current_action], " ", self.p_a_mb[self.current_action]
        # print "tmp =", tmp
        ninf = np.isinf(tmp).sum()        

        if np.isinf(tmp).sum():            
            self.p_a_final = np.isinf(tmp)*((1.0/float(ninf))-ninf*0.0000001-0.0000001/ninf) + 0.0000001
        else :
            self.p_a_final = tmp/np.sum(tmp)   
        
        if np.sum(self.p_a_final == 0.0):
            self.p_a_final+=1e-8;
            self.p_a_final = self.p_a_final/self.p_a_final.sum()        

    def updateValue(self, reward):
        # print "R = ", reward
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
        # Updating model free
        r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0                               
        self.delta = float(r)-self.values_mf[self.current_action]        
        self.values_mf[self.current_action] = self.values_mf[self.current_action]+self.parameters['alpha']*self.delta                
        index = range(self.n_action)
        index.pop(int(self.current_action))        
        self.values_mf[index] = self.values_mf[index] + (1.0-self.parameters['kappa']) * (0.0 - self.values_mf[index])    

class CSelection():
    """ mixture strategy
    """
    def __init__(self):
        self.n_action = 4
        self.n_r = 2
        self.max_entropy = -np.log2(1./self.n_action)
        self.bounds = dict({"length":[1, 10], 
                    "threshold":[0.01, self.max_entropy], 
                    "noise":[0.0, 0.1],                            
                    'alpha':[0.0, 1.0],
                    "beta":[0.0, 100.0], # QLEARNING
                    "sigma":[0.0, 20.0], 
                    "weight":[0.0, 1.0], 
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
        self.p_a = np.zeros((int(self.parameters['length']), self.n_action))
        self.p_r_a = np.zeros((int(self.parameters['length']), self.n_action, 2))
        self.nb_inferences = 0
        self.n_element = 0
        self.Hb = self.max_entropy        
        self.uniform = np.ones((self.n_action, 2))*(1./(self.n_action*2))
        self.problem = self.sari[0,1]
        self.p_a_final = np.zeros(self.n_action)
        self.spatial_biases = np.ones(self.n_action) * (1./self.n_action)        
        self.w = self.parameters['weight']
        for i in xrange(self.N):
        # for i in xrange(9):
            if self.sari[i][1] != self.problem:
                if self.sari[i][4]-self.sari[i-1][4] < 0.0:
                    # START BLOC
                    self.problem = self.sari[i][1]
                    self.n_element = 0
                    self.w = self.parameters['weight']
                    # RESET Q-LEARNING SPATIAL BIASES AND REWARD SHIFT
                    # print "biais", self.spatial_biases
                    self.values_mf = self.spatial_biases/self.spatial_biases.sum()
                    # shift bias
                    tmp = self.values_mf[self.current_action]
                    self.values_mf *= self.parameters['shift']/3.
                    self.values_mf[self.current_action] = tmp*(1.0-self.parameters['shift'])
                    # spatial biaises
                    self.spatial_biases[self.sari[i,2]-1] += 1.0

            # START TRIAL
            self.current_action = self.sari[i][2]-1
            # print "PROBLEM=", self.problem, " ACTION=", self.current_action
            r = self.sari[i][0]            
                        
            # BAYESIAN CALL
            self.p = self.uniform[:,:]
            self.Hb = self.max_entropy
            self.nb_inferences = 0  
            self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)        

            while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:            
                self.inferenceModule()
                self.evaluationModule()                    

            self.fusionModule()
            self.value[i] = float(np.log(self.p_a_final[self.current_action])) 
            # print self.value[i]
            H = -(self.p_a_final*np.log2(self.p_a_final)).sum()
            self.reaction[i] = float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma'])+H)
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

    def analysis_call(self, sari, mean_rt, parameters):        
        self.parameters = parameters
        self.sari = sari[:,[3,4,5,9,2]] # reward | problem | action | index | phase
        self.N = len(self.sari)        
        self.mean_rt = mean_rt
        self.value = np.zeros(self.N)
        self.reaction = np.zeros(self.N)
        self.values_mf =  np.zeros(self.n_action)
        self.p_a = np.zeros((int(self.parameters['length']), self.n_action))
        self.p_r_a = np.zeros((int(self.parameters['length']), self.n_action, 2))
        self.nb_inferences = 0
        self.n_element = 0
        self.Hb = self.max_entropy        
        self.uniform = np.ones((self.n_action, 2))*(1./(self.n_action*2))
        self.problem = self.sari[0,1]
        self.p_a_final = np.zeros(self.n_action)
        self.spatial_biases = np.ones(self.n_action) * (1./self.n_action)        
        self.w = self.parameters['weight']
        ## LIST ####
        self.w_list = np.zeros(self.N)
        self.entropy_list = np.zeros((self.N,2))
        self.free_list = np.zeros((self.N,4))
        self.biais_list = np.zeros((self.N,4))
        self.delta_list = np.zeros((self.N,4))
        ############
        for i in xrange(self.N):        
            if self.sari[i][1] != self.problem:
                if self.sari[i][4]-self.sari[i-1][4] < 0.0:
                    # START BLOC
                    self.problem = self.sari[i][1]
                    self.n_element = 0
                    self.w = self.parameters['weight']
                    # RESET Q-LEARNING SPATIAL BIASES AND REWARD SHIFT                    
                    self.values_mf = self.spatial_biases/self.spatial_biases.sum()
                    # shift bias
                    tmp = self.values_mf[self.current_action]
                    self.values_mf *= self.parameters['shift']/3.
                    self.values_mf[self.current_action] = tmp*(1.0-self.parameters['shift'])
                    # spatial biaises update
                    self.spatial_biases[self.sari[i,2]-1] += 1.0

            # START TRIAL
            self.current_action = self.sari[i][2]-1            
            r = self.sari[i][0]            
            

            # BAYESIAN CALL
            self.p = self.uniform[:,:]
            self.Hb = self.max_entropy
            self.nb_inferences = 0  
            self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)        

            while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:            
                self.inferenceModule()
                self.evaluationModule()                    

            self.fusionModule()

            self.w_list[i] = self.w
            self.entropy_list[i,0] = self.Hb
            self.entropy_list[i,1] = self.Hf
            self.free_list[i] = self.values_mf
            self.biais_list[i] = self.spatial_biases

            self.value[i] = float(np.log(self.p_a_final[self.current_action])) 
            # print self.value[i]
            H = -(self.p_a_final*np.log2(self.p_a_final)).sum()
            self.reaction[i] = float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma'])+H)
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

    def fusionModule(self):
        np.seterr(invalid='ignore')
        self.p_a_mf = np.exp(self.values_mf*float(self.parameters['beta']))
        self.p_a_mf = self.p_a_mf/np.sum(self.p_a_mf)
        self.Hf = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
        
        self.p_a_final = (1.0-self.w)*self.p_a_mf + self.w*self.p_a_mb                
        self.q_values = self.p_a_final      

        #nombre de inf
        ninf = np.isinf(self.p_a_final).sum()  
        if np.isinf(self.p_a_final).sum():
            self.p_a_final = np.isinf(tmp)*((1.0/float(ninf))-ninf*0.0000001-0.0000001/ninf) + 0.0000001
        else :
            self.p_a_final = self.p_a_final/np.sum(self.p_a_final)   
        
        if np.sum(self.p_a_final == 0.0):
            self.p_a_final+=1e-8;
            self.p_a_final = self.p_a_final/self.p_a_final.sum()
                    
    def updateWeight(self, r):
        if r:
            p_wmc = self.p_a_mb[self.current_action]
            p_rl = self.p_a_mf[self.current_action]
        else:
            p_wmc = 1.0 - self.p_a_mb[self.current_action]
            p_rl = 1.0 - self.p_a_mf[self.current_action]
        self.w = (p_wmc*self.w)/(p_wmc*self.w + p_rl * (1.0 - self.w))
        # self.p_wm[-1].append(self.p_a_mb[self.current_action])
        # self.p_rl[-1].append(self.p_a_mf[self.current_action])    

    def updateValue(self, reward):
        r = int((reward==1)*1)
        # Specific to Collins model
        self.updateWeight(float(r))
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
        # Updating model free
        r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0                        
        self.delta = float(r)-self.values_mf[self.current_action]        
        self.values_mf[self.current_action] = self.values_mf[self.current_action]+self.parameters['alpha']*self.delta                
        index = range(self.n_action)
        index.pop(int(self.current_action))        
        self.values_mf[index] = self.values_mf[index] + (1.0-self.parameters['kappa']) * (0.0 - self.values_mf[index])        


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
        # for i in xrange(709):            
            if self.sari[i][1] != self.problem:
                if self.sari[i][4]-self.sari[i-1][4] < 0.0:
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
            self.reaction[i] = float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma'])+H)
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


class QLearning():
    """ mixture strategy
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
        self.reaction = np.zeros(self.n_action)
        self.values_mf =  np.zeros(self.n_action)
        self.problem = self.sari[0,1]
        self.p_a_final = np.zeros(self.n_action)
        self.spatial_biases = np.ones(self.n_action) * (1./self.n_action)        
        for i in xrange(self.N):
            if self.sari[i][1] != self.problem:
                if self.sari[i][4]-self.sari[i-1][4] < 0.0:
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
            self.reaction[i] = float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma'])+H)
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













class KSelection():
    """Class that implement Keramati models for action selection
    Specially tuned for Brovelli experiment so beware
    """
    def __init__(self, states, actions, parameters={"length":1,"eta":0.0001}, var_obs = 0.05, init_cov = 10, kappa = 0.1, sferes=False):
        #State Action Spaces
        self.states=states
        self.actions=actions
        #Parameters
        self.sferes = sferes
        self.parameters = parameters
        self.n_action = int(len(actions))
        self.n_state = int(len(states))
        self.bounds = dict({"beta":[0.0, 100.0],
                            "eta":[0.00001, 0.001],
                            "length":[1, 10],
                            "threshold":[0.01, -np.log2(1./self.n_action)], 
                            "noise":[0.0,0.1],
                            "sigma":[0.0,1.0],
                            "sigma_rt":[0.0, 20.0]})
                            #"sigma_ql":[0.00001, 1.0]})        
        self.var_obs = var_obs
        self.init_cov = init_cov
        self.kappa = kappa
        #Probability Initialization
        self.uniform = np.ones((self.n_state, self.n_action, 2))*(1./(self.n_state*self.n_action*2))
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.p_a_mf = None
        self.p_a_mb = None
        self.p = None
        self.p_a = None
        self.pA = None
        # QValues model free
        self.values_mf = np.zeros((self.n_state, self.n_action))
        self.covariance = createCovarianceDict(self.n_state*self.n_action, self.init_cov, self.parameters['eta'])
        self.point = None
        self.weights = None
        # Control initialization
        self.nb_inferences = 0
        self.n_element= 0
        self.current_state = None
        self.current_action = None
        self.max_entropy = -np.log2(1./self.n_action)
        self.Hb = self.max_entropy
        self.Hf = self.max_entropy
        self.Hl = self.max_entropy
        self.N = 0
        self.q_values = np.zeros(self.n_action)
        self.delta = 0.0        
        self.reward_rate = np.zeros(self.n_state)
        # List Init
        if self.sferes:
            self.value = np.zeros((n_blocs, n_trials))
            self.reaction = np.zeros((n_blocs, n_trials))
        else:
            self.state = list()
            self.action = list()
            self.responses = list()
            self.reaction = list()
            self.value = list()
            self.vpi = list()
            self.rrate = list()
            self.Hall = list()
            self.pdf = list()
            #self.sigma = list()
            #self.sigma_test = list()        

    def setParameters(self, name, value):            
        if value < self.bounds[name][0]:
            self.parameters[name] = self.bounds[name][0]
        elif value > self.bounds[name][1]:
            self.parameters[name] = self.bounds[name][1]
        else:
            self.parameters[name] = value                

    def setAllParameters(self, parameters):
        for i in parameters.iterkeys():
            if i in self.bounds.keys():
                self.setParameters(i, parameters[i])

    def startBloc(self):
        if not self.sferes:
            self.state.append([])
            self.action.append([])
            self.responses.append([])
            self.reaction.append([])
            self.vpi.append([])
            self.rrate.append([])
            #self.sigma_test.append([])
            self.Hall.append([])
            self.pdf.append([])
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.values_mf = np.zeros((self.n_state, self.n_action))
        self.covariance = createCovarianceDict(self.n_state*self.n_action, self.init_cov, self.parameters['eta'])
        self.reward_rate = np.zeros(self.n_state)        
        self.nb_inferences = 0
        self.n_element = 0
        self.values = None
        self.current_state = None
        self.current_action = None
        self.Hb = self.max_entropy
        self.Hf = self.max_entropy

    def startExp(self):                
        self.state = list()
        self.action = list()
        self.responses = list()
        self.reaction = list()
        self.value = list()   
        self.vpi = list()
        self.rrate = list() 
        #self.sigma = list()    
        #self.sigma_test = list()
        self.pdf = list()
        self.Hall = list()
        self.pdf = list()

    def sampleSoftMax(self, values):
        tmp = np.exp(values*float(self.parameters['beta']))
        tmp = tmp/float(np.sum(tmp))
        tmp = [np.sum(tmp[0:i]) for i in range(len(tmp))]
        return np.sum(np.array(tmp) < np.random.rand())-1        

    def inferenceModule(self):        
        tmp = self.p_a_s[self.nb_inferences] * np.vstack(self.p_s[self.nb_inferences])
        self.p = self.p + self.p_r_as[self.nb_inferences] * np.reshape(np.repeat(tmp, 2, axis = 1), self.p_r_as[self.nb_inferences].shape)
        self.nb_inferences+=1

    def evaluationModule(self):
        tmp = self.p/np.sum(self.p)
        p_ra_s = tmp[self.current_state]/np.sum(tmp[self.current_state])
        p_r_s = np.sum(p_ra_s, axis = 0)
        p_a_rs = p_ra_s/p_r_s
        self.p_a_mb = p_a_rs[:,1]/p_a_rs[:,0]
        p_a_mb = self.p_a_mb/np.sum(self.p_a_mb)
        self.Hb = -np.sum(p_a_mb*np.log2(p_a_mb))
        self.values = p_a_rs[:,1]/p_a_rs[:,0]
        self.values = self.values/np.sum(self.values)

    def predictionStep(self):
        self.covariance['noise'] = self.covariance['cov']*self.parameters['eta']        
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] + self.covariance['noise']    

    def computeSigmaPoints(self):        
        n = self.n_state*self.n_action
        self.point = np.zeros((2*n+1, n))
        self.point[0] = self.values_mf.flatten()
        c = np.linalg.cholesky((n+self.kappa)*self.covariance['cov'])        
        self.point[range(1,n+1)] = self.values_mf.flatten()+np.transpose(c)
        self.point[range(n+1, 2*n+1)] = self.values_mf.flatten()-np.transpose(c)
        # print np.array2string(self.point, precision=2, separator='',suppress_small=True)
        self.weights = np.zeros((2*n+1,1))
        self.weights[1:2*n+1] = 1/(2*n+self.kappa)

    def updateRewardRate(self, reward, delay = 0.0):
        self.reward_rate[self.current_state] = (1.0-self.parameters['sigma'])*self.reward_rate[self.current_state]+self.parameters['sigma']*reward
        if not self.sferes:        
            self.rrate[-1].append(self.reward_rate[self.current_state])

    def softMax(self, values):
        tmp = np.exp(values*float(self.parameters['beta']))
        if np.isinf(tmp).sum():
            self.p_a = np.isinf(self.p_a)*0.9999995+0.0000001
        else :
            self.p_a = tmp/np.sum(tmp)           
        return tmp/float(np.sum(tmp))

    def sample(self, values):
        tmp = [np.sum(values[0:i]) for i in range(len(values))]
        return np.sum(np.array(tmp) < np.random.rand())-1        
        
    def computeValue(self, s, a, ind):
        self.current_state = s
        self.current_action = a        
        self.nb_inferences = 0
        self.predictionStep()
        self.q_values = self.values_mf[self.current_state]        
        self.p_a = self.softMax(self.values_mf[self.current_state])        
        self.Hf = -(self.p_a*np.log2(self.p_a)).sum()       
        t = self.n_action*self.current_state
        self.vpi = computeVPIValues(self.values_mf[self.current_state], self.covariance['cov'].diagonal()[t:t+self.n_action])
        self.r_rate = self.reward_rate[self.current_state]        
        if np.sum(self.vpi > self.reward_rate[self.current_state]):                
            self.p = self.uniform[:,:]
            self.Hb = self.max_entropy            
            self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)
            while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:
                self.inferenceModule()
                self.evaluationModule()
            self.q_values = self.p_a_mb
            self.p_a = self.p_a_mb/np.sum(self.p_a_mb)
        

        H = -(self.p_a*np.log2(self.p_a)).sum()
        self.N = self.nb_inferences

        # if np.isnan(values).sum(): values = np.isnan(values)*0.9995+0.0001            
        # if np.isnan(H): H = 0.005
        self.value[ind] = float(np.log(self.p_a[self.current_action]))
        # self.reaction[ind] = float(H*self.parameters['sigma_rt']+np.log2(N))
        self.reaction[ind] = float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma_rt'])+H)
        
        
    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.nb_inferences = 0
        self.predictionStep()
        values = self.softMax(self.values_mf[self.current_state])
        self.Hf = -(values*np.log2(values)).sum()       
        t =self.n_action*self.current_state
        vpi = computeVPIValues(self.values_mf[self.current_state], self.covariance['cov'].diagonal()[t:t+self.n_action])
        
        self.used = -1
        if np.sum(vpi > self.reward_rate[self.current_state]):
            self.used = 1
            self.p = self.uniform[:,:]
            self.Hb = self.max_entropy            
            self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)
            while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:
                self.inferenceModule()
                self.evaluationModule()

            values = self.p_a_mb/np.sum(self.p_a_mb)
        self.current_action = self.sample(values)
        self.value.append(float(values[self.current_action]))
        self.action[-1].append(self.current_action)
        H = -(values*np.log2(values)).sum()
        N = float(self.nb_inferences+1)
        self.Hl = H        
        self.reaction[-1].append((np.log2(N)**self.parameters['sigma_rt'])+H)

        self.Hall[-1].append([float(self.Hb), float(self.Hf)])
        self.pdf[-1].append(N)

        self.vpi[-1].append(vpi[self.current_action])        
        # qlearning        
        self.h_ql_only = self.Hf
        # bayesian            
        self.h_bayes_only = self.Hb

        return self.actions[self.current_action]


    def updateValue(self, reward):
        r = int((reward==1)*1)
        if not self.sferes:
            self.responses[-1].append(r)
        if self.parameters['noise']:
            self.p_s = self.p_s*(1-self.parameters['noise'])+self.parameters['noise']*(1.0/self.n_state*np.ones(self.p_s.shape))
            self.p_a_s = self.p_a_s*(1-self.parameters['noise'])+self.parameters['noise']*(1.0/self.n_action*np.ones(self.p_a_s.shape))
            self.p_r_as = self.p_r_as*(1-self.parameters['noise'])+self.parameters['noise']*(0.5*np.ones(self.p_r_as.shape))
        #Shifting memory            
        if self.n_element < int(self.parameters['length']):
            self.n_element+=1
        self.p_s[1:self.n_element] = self.p_s[0:self.n_element-1]
        self.p_a_s[1:self.n_element] = self.p_a_s[0:self.n_element-1]
        self.p_r_as[1:self.n_element] = self.p_r_as[0:self.n_element-1]
        self.p_s[0] = 0.0
        self.p_a_s[0] = np.ones((self.n_state, self.n_action))*(1/float(self.n_action))
        self.p_r_as[0] = np.ones((self.n_state, self.n_action, 2))*0.5
        #Adding last choice                 
        self.p_s[0, self.current_state] = 1.0        
        self.p_a_s[0, self.current_state] = 0.0
        self.p_a_s[0, self.current_state, self.current_action] = 1.0
        self.p_r_as[0, self.current_state, self.current_action] = 0.0
        self.p_r_as[0, self.current_state, self.current_action, int(r)] = 1.0        
        # Updating model free
        r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0        
        self.computeSigmaPoints()                        
        t =self.n_action*self.current_state+self.current_action
        # rewards_predicted = (self.point[:,t]-self.parameters['gamma']*np.max(self.point[:,self.n_action*self.current_state:self.n_action*self.current_state+self.n_action], 1)).reshape(len(self.point), 1)
        rewards_predicted = (self.point[:,t]).reshape(len(self.point), 1)                
        reward_predicted = np.dot(rewards_predicted.flatten(), self.weights.flatten())                
        cov_values_rewards = np.sum(self.weights*(self.point-self.values_mf.flatten())*(rewards_predicted-reward_predicted), 0)        
        cov_rewards = np.sum(self.weights*(rewards_predicted-reward_predicted)**2) + self.var_obs        
        kalman_gain = cov_values_rewards/cov_rewards 
        self.delta = ((kalman_gain*(r-reward_predicted)).reshape(self.n_state, self.n_action))[self.current_state]
        self.values_mf = (self.values_mf.flatten() + kalman_gain*(r-reward_predicted)).reshape(self.n_state, self.n_action)        
        self.covariance['cov'][:,:] = self.covariance['cov'][:,:] - (kalman_gain.reshape(len(kalman_gain), 1)*cov_rewards)*kalman_gain
        # Updating selection 
        self.updateRewardRate(r)


            
