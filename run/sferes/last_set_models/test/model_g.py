#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from time import sleep
from scipy.stats import sem

class CSelection_g_2():
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
            if self.sari[i][4]-self.sari[i-1][4] < 0.0 and i > 0:
                    # START BLOC
                    self.problem = self.sari[i][1]
                    self.n_element = 0
                    self.w = self.parameters['weight']
                    # RESET Q-LEARNING SPATIAL BIASES AND REWARD SHIFT
                    # print "biais", self.spatial_biases
                    self.values_mf = np.zeros(4)
                    #self.values_mf = self.spatial_biases/self.spatial_biases.sum()
                    # shift bias
                    #tmp = self.values_mf[self.current_action]
                    #self.values_mf *= self.parameters['shift']/3.
                    #self.values_mf[self.current_action] = tmp*(1.0-self.parameters['shift'])
                    # spatial biaises
                    #self.spatial_biases[self.sari[i,2]-1] += 1.0

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
            # self.reaction[i] = self.Hb + self.parameters['sigma']*self.Hf            
            
            self.updateValue(r)

        # ALIGN TO MEDIAN
        self.reaction = self.reaction - np.median(self.reaction)
        self.reaction = self.reaction / (np.percentile(self.reaction, 75)-np.percentile(self.reaction, 25))        
        # LEAST SQUARES            
        self.rt_model = np.zeros((len(self.mean_rt),2))
        for i in xrange(len(self.rt_model)):
            self.rt_model[i,0] = np.mean(self.reaction[self.sari[:,3] == i])
            self.rt_model[i,1] = np.var(self.reaction[self.sari[:,3] == i])

        # FIT
        fit = np.zeros(2)
        fit[0] = np.sum(self.value)
        fit[1] = -np.sum(np.power(self.rt_model[:,0]-self.mean_rt[:,1], 2.0))
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
        self.based_list = np.zeros((self.N,4))
        self.biais_list = np.zeros((self.N,4))
        self.delta_list = np.zeros((self.N,4))
        self.inference_list = np.zeros((self.N,1))
        self.wmean_dict = {i:np.zeros(i+3) for i in xrange(1,6)}
        self.wmean_count = {i:np.zeros(i+3) for i in xrange(1,6)}
        start = 0
        ############
        
        nb_problems = 0
        nb_non_suivi = 0 
        for i in xrange(self.N):                       
            if self.sari[i][4]-self.sari[i-1][4] < 0.0 and i > 0:
                    nb_problems += 1
                    if self.sari[i][1] != self.problem:
                        nb_non_suivi += 1                        
                    # START BLOC
                    # print "bloc ", i
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
                    #######################
                    nb_search = np.sum(self.sari[start:i-1,4] == 0)
                    nb_repeat = np.sum(self.sari[start:i-1,4] == 1)
                    if nb_repeat >= 3 and nb_search <= 5:
                        self.wmean_dict[nb_search] += self.w_list[start:start+nb_search+3]
                        self.wmean_count[nb_search] += 1.0
                    elif nb_repeat < 3 and nb_search <= 5:
                        self.wmean_dict[nb_search][0:nb_search+nb_repeat] += self.w_list[start:start+nb_search+nb_repeat]
                        self.wmean_count[nb_search][0:nb_search+nb_repeat] += 1.0
                    start = i
                    #######################

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

            ##########################
            self.w_list[i] = self.w
            self.entropy_list[i,0] = self.Hb
            self.entropy_list[i,1] = self.Hf
            self.free_list[i] = self.p_a_mf
            self.based_list[i] = self.p_a_mb
            self.biais_list[i] = self.spatial_biases            
            self.inference_list[i] = self.nb_inferences            
            ##########################

            self.value[i] = float(np.log(self.p_a_final[self.current_action])) 
            # print self.value[i]
            H = -(self.p_a_final*np.log2(self.p_a_final)).sum()
            self.reaction[i] = float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma'])+H)
            # self.reaction[i] = self.Hb + self.parameters['sigma']*self.Hf
            # print self.reaction[i]
            self.updateValue(r)
            

        # ALIGN TO MEDIAN
        self.reaction = self.reaction - np.median(self.reaction)
        self.reaction = self.reaction / (np.percentile(self.reaction, 75)-np.percentile(self.reaction, 25))        
        # LEAST SQUARES            
        self.rt_model = np.zeros(len(self.mean_rt))
        for i in xrange(len(self.rt_model)):
            self.rt_model[i] = np.mean(self.reaction[self.sari[:,3] == i])
        
        ##########################
        for k in self.wmean_dict.iterkeys():
            self.wmean_dict[k] = self.wmean_dict[k]/self.wmean_count[k]
        ##########################

    def test_call(self, sari, mean_rt, parameters, N = 100000):        
        self.parameters = parameters
        self.sari = sari[:,[3,4,5,9,2]] # reward | problem | action | index | phase
        self.N = len(self.sari)        
        self.mean_rt = mean_rt
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
        self.reaction = np.zeros(N)

        ## LIST ####
        self.performance = np.zeros(30)
        self.count = np.zeros(30)
        self.reaction = [] # pas le choix
        self.index = [] 
        ############
        correct = 0        
        for i in xrange(N):   # pour chaque probleme
            # nouveau problem            
            self.n_element = 0
            self.values_mf = np.zeros(4)
            self.w = self.parameters['weight']
            reward = 0
            count = 0
            p = np.ones(4)*0.3
            p[correct] = 0.1
            correct = np.random.choice(np.arange(4), p = p)
            # phase de search             
            for j in xrange(5):
                # trial
                self.p = self.uniform[:,:]
                self.Hb = self.max_entropy
                self.nb_inferences = 0  
                self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)        
                while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:            
                    self.inferenceModule()
                    self.evaluationModule()                    
                self.fusionModule()                
                self.current_action = self.sample(self.p_a_final)
                H = -(self.p_a_final*np.log2(self.p_a_final)).sum()
                self.reaction.append(float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma'])+H))                

                if self.current_action == correct:
                    reward = 1
                    self.updateValue(reward)
                    break
                else :
                    reward = 0
                    self.updateValue(reward)
            type_de_problem = j
            # phase de repeat
            for k in xrange(3):            
                # trial
                self.p = self.uniform[:,:]
                self.Hb = self.max_entropy
                self.nb_inferences = 0  
                self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)        
                while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:            
                    self.inferenceModule()
                    self.evaluationModule()                    
                self.fusionModule()                
                self.current_action = self.sample(self.p_a_final)
                H = -(self.p_a_final*np.log2(self.p_a_final)).sum()
                self.reaction.append(float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma'])+H))                
                if self.current_action == correct:
                    reward = 1                    
                else :
                    reward = 0
                self.updateValue(reward)

            start = 3*(type_de_problem)+ np.arange(1,type_de_problem+1).sum()
            stop = start+type_de_problem+1+3
            for l in xrange(start, stop):
                self.index.append(l)

        self.index = np.array(self.index)
        self.reaction = np.array(self.reaction)
        # # ALIGN TO MEDIAN
        self.reaction = self.reaction - np.median(self.reaction)
        self.reaction = self.reaction / (np.percentile(self.reaction, 75)-np.percentile(self.reaction, 25))        
        # # LEAST SQUARES            
        self.rt_model = np.zeros(self.mean_rt.shape)
        for i in xrange(len(self.rt_model)):
            self.rt_model[i,0] = self.mean_rt[i,0]
            self.rt_model[i,1] = np.mean(self.reaction[self.index == i])
            # self.rt_model[i,2] = np.var(self.reaction[self.index == i])
            self.rt_model[i,2] = sem(self.reaction[self.index == i])

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
        tmp0 = self.values_mf - np.max(self.values_mf)
        self.p_a_mf = np.exp(tmp0*float(self.parameters['beta']))
        self.p_a_mf = self.p_a_mf/np.sum(self.p_a_mf)
        self.Hf = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
        
        self.p_a_final = (1.0-self.w)*self.p_a_mf + self.w*self.p_a_mb                
        self.q_values = self.p_a_final      

        #nombre de inf
        ninf = np.isinf(self.p_a_final).sum()  
        if ninf:
            print "INF"
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
        self.delta = float(r) + self.parameters['shift']*np.max(self.values_mf) - self.values_mf[self.current_action]        
        # if r == 1:
        #     self.values_mf[self.current_action] = self.values_mf[self.current_action]+self.parameters['alphap']*self.delta                
        # elif r == 0 or r == -1:
        #     self.values_mf[self.current_action] = self.values_mf[self.current_action]+self.parameters['alpham']*self.delta                        
        self.values_mf[self.current_action] = self.values_mf[self.current_action]+self.parameters['alpha']*self.delta                        
        # index = range(self.n_action)
        # index.pop(int(self.current_action))        
        # self.values_mf[index] = self.values_mf[index] + (1.0-self.parameters['kappa']) * (0.0 - self.values_mf[index])        
