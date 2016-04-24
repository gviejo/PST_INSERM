#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import numpy as np
import sys

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
                            "threshold":[0.00001, 1000.0], # sigmoide parameter
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
        # for i in xrange(21):              
            if self.sari[i][1] != self.problem:                
                if self.sari[i][4]-self.sari[i-1][4] < 0.0:                    
                    # START BLOC
                    self.problem = self.sari[i][1]
                    self.n_element = 0
                    # RESET Q-LEARNING SPATIAL BIASES AND REWARD SHIFT
                    # print "biais", self.spatial_biases
                    self.values_mf = self.spatial_biases/self.spatial_biases.sum()
                    self.values_mf[self.current_action] *= (1.0-self.parameters['shift'])
                    self.spatial_biases[self.sari[i,2]-1] += 1.0


            # START TRIAL
            self.current_action = self.sari[i][2]-1
            # print "PROBLEM=", self.problem, " ACTION=", self.current_action
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
            # print 0, " p_ak=", self.p_ak[0], " p_decision=", self.p_decision[0] , " p_retrieval=", self.p_retrieval[0]      
            # print "N element =", self.n_element
            for j in xrange(self.n_element):            
                self.inferenceModule()
                self.evaluationModule()
                # print "mf = ", self.values_mf
                self.fusionModule()                
                self.p_ak[j+1] = self.p_a_final[self.current_action]                
                H = -(self.p_a_final*np.log2(self.p_a_final)).sum()
                N = self.nb_inferences+1.0
                reaction[j+1] = float(((np.log2(N))**self.parameters['sigma'])+H)
                self.sigmoideModule()
                self.p_sigmoide[j+1] = self.pA            
                self.p_decision[j+1] = self.pA*self.p_retrieval[j]            
                self.p_retrieval[j+1] = (1.0-self.pA)*self.p_retrieval[j]                    
                # print j+1, " p_ak=", self.p_ak[j+1], " p_decision=", self.p_decision[j+1], " p_retrieval=", self.p_retrieval[0]
            
            # print np.dot(self.p_decision,self.p_ak)
            self.value[i] = float(np.log(np.dot(self.p_decision,self.p_ak)))        
            # print self.value[i]
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
        # print "p_afinal =", self.p_a_final
        # if not self.sferes:
        #     # qlearning
        #     tmp = np.exp(self.values_mf[self.current_state]*float(self.parameters['beta']))
        #     pa = tmp/np.sum(tmp)
        #     # print "Q_ql("+str(self.current_state)+")=",self.values_mf[self.current_state]            
        #     # print "p_ql("+str(self.current_state)+")=",pa            
        #     self.h_ql_only = -np.sum(pa*np.log2(pa))
        #     # print "H_ql =", self.h_ql_only
        #     # bayesian
        #     tmp = np.exp(self.p_a_mb*float(self.parameters['beta']))
        #     pa = tmp/np.sum(tmp)
        #     # print "N=", self.nb_inferences
        #     # print "Q_ba("+str(self.current_state)+")=",self.p_a_mb
        #     # print "p_ba("+str(self.current_state)+")=",pa
        #     self.h_bayes_only = -np.sum(pa*np.log2(pa))
        #     # print "H_ba =", self.h_bayes_only
        
    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.p = self.uniform[:,:,:]
        self.Hb = self.max_entropy
        self.p_a_mf = SoftMaxValues(self.values_mf[self.current_state], self.parameters['gamma'])
        self.Hf = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
        self.nb_inferences = 0
        self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)
        
        # print "Q("+state+")= ", self.values_mf[self.current_state]
        # print "p_a_mf("+state+")=",self.p_a_mf

        while self.sigmoideModule():
            self.inferenceModule()
            self.evaluationModule()

        self.fusionModule()
        self.current_action = self.sample(self.p_a)
        self.value.append(float(self.p_a[self.current_action]))
        self.action[-1].append(self.current_action)                
        self.Hall[-1].append([float(self.Hb), float(self.Hf)])
        H = -(self.p_a*np.log2(self.p_a)).sum()
        self.Hl = H
        # if np.isnan(H): H = 0.005                        
        N = float(self.nb_inferences+1)        
        # self.reaction[-1].append(float(H*self.parameters['sigma']+np.log2(N)))                
        # self.reaction[-1].append(float((N**self.parameters['sigma'])+H))
        # self.reaction[-1].append(float(self.parameters['sigma']*np.log2(N)+H))
        # self.reaction[-1].append(float((np.log2(N)+H)**self.parameters['sigma']))
        self.reaction[-1].append(float(((np.log2(N))**self.parameters['sigma'])+H))
        self.pdf[-1].append(N)

        # self.reaction[-1].append(N-1)
        
        return self.actions[self.current_action]

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
        # print "R = ", float(r)
        # print self.values_mf[self.current_action]        
        self.delta = float(r)-self.values_mf[self.current_action]        
        # print "delta = ", self.delta
        self.values_mf[self.current_action] = self.values_mf[self.current_action]+self.parameters['alpha']*self.delta        
        # print " mf2=" , self.values_mf
        index = range(self.n_action)
        index.pop(int(self.current_action))        
        self.values_mf[index] = self.values_mf[index] + (1.0-self.parameters['kappa']) * ((1.0/self.n_action) - self.values_mf[index])
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
            self.p = self.uniform[:,:,:]
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
            self.p = self.uniform[:,:,:]
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


            
class CSelection():
    """Class that implement Collins models for action selection
    Model-based must be provided
    Specially tuned for Brovelli experiment so beware
    """
    def __init__(self, states, actions, parameters={'length':1, 'weight':0.5}, sferes = False):
        # State Action Space        
        self.states=states
        self.actions=actions        
        #Parameters
        self.sferes = sferes
        self.parameters = parameters
        self.n_action=int(len(actions))
        self.n_state=int(len(states))
        self.max_entropy = -np.log2(1./self.n_action)
        self.bounds = dict({"length":[1, 10], 
                            "threshold":[0.01, self.max_entropy], 
                            "noise":[0.0, 0.1],                            
                            'alpha':[0.0, 1.0],
                            "beta":[0.0, 100.0], # QLEARNING
                            "sigma":[0.0, 20.0], 
                            "weight":[0.0, 1.0]})
                            
                            

        # Probability Initialization        
        self.uniform = np.ones((self.n_state, self.n_action, 2))*(1./(self.n_state*self.n_action*2))
        self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)    
        self.p = None        
        self.p_a = None
        # Specific to collins        
        self.w = np.ones(self.n_state)*self.parameters['weight']
        self.q_mb = np.zeros((self.n_action))
        # Q-values model free
        self.q_mf = np.zeros((self.n_state, self.n_action))
        self.p_a_mf = None
        # Various Init
        self.nb_inferences = 0
        self.current_state = None
        self.current_action = None        
        self.entropy = self.max_entropy        
        self.n_element = 0
        self.q_values = np.zeros(self.n_action)
        self.Hb = 0.0
        self.Hf = 0.0
        self.N = 0
        self.delta = 0.0        
        # Optimization init
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.p_r_s = np.ones(2)*0.5
        #List Init
        if self.sferes:
            self.value = np.zeros((n_blocs, n_trials))
            self.reaction = np.zeros((n_blocs, n_trials))
        else:
            self.state=list()        
            self.action=list()
            self.responses=list()        
            self.reaction=list()
            self.value=list()
            self.pdf = list()
            self.Hall = list()
            self.pdf = list()


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
            self.weights.append([])
            self.p_wm.append([])
            self.p_rl.append([])
            self.Hall.append([])
            self.pdf.append([])
        self.n_element = 0
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.p_a = np.ones(self.n_action)*(1./self.n_action)        
        self.w = np.ones(self.n_state)*self.parameters['weight']
        self.q_mb = np.zeros((self.n_action))
        self.q_mf = np.zeros((self.n_state, self.n_action))
        self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)    
        self.nb_inferences = 0
        self.current_state = None
        self.current_action = None

    def startExp(self):
        self.n_element = 0
        self.p_s = np.zeros((int(self.parameters['length']), self.n_state))
        self.p_a_s = np.zeros((int(self.parameters['length']), self.n_state, self.n_action))
        self.p_r_as = np.zeros((int(self.parameters['length']), self.n_state, self.n_action, 2))
        self.state=list()
        self.action=list()
        self.reaction=list()
        self.responses=list()
        self.value=list()
        self.p_a = np.ones(self.n_action)*(1./self.n_action)        
        self.weights=list()
        self.p_wm=list()
        self.p_rl=list()
        self.Hall=list()
        self.pdf=list()

    def sample(self, values):
        tmp = [np.sum(values[0:i]) for i in range(len(values))]
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
        self.q_mb = p_a_rs[:,1]/p_a_rs[:,0]        
        # self.p_a_mb = np.exp(self.q_mb*float(self.parameters['gain']))        
        self.p_a_mb = self.q_mb/np.sum(self.q_mb)
        # self.p_a_mb = self.p_a_mb/np.sum(self.p_a_mb)        
        self.Hb = -np.sum(self.p_a_mb*np.log2(self.p_a_mb))
        

    def fusionModule(self):
        np.seterr(invalid='ignore')
        self.p_a_mf = np.exp(self.q_mf[self.current_state]*float(self.parameters['beta']))
        self.p_a_mf = self.p_a_mf/np.sum(self.p_a_mf)
        self.Hf = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
        self.p_a = (1.0-self.w[self.current_state])*self.p_a_mf + self.w[self.current_state]*self.p_a_mb                
        self.q_values = self.p_a      

        #nombre de inf
        ninf = np.isinf(self.p_a).sum()  
        if np.isinf(self.p_a).sum():
            self.p_a = np.isinf(tmp)*((1.0/float(ninf))-ninf*0.0000001-0.0000001/ninf) + 0.0000001
        else :
            self.p_a = self.p_a/np.sum(self.p_a)   
        
        if not self.sferes:
            # qlearning
            tmp = np.exp(self.q_mf[self.current_state]*float(self.parameters['beta']))
            pa = tmp/np.sum(tmp)
            self.h_ql_only = -np.sum(pa*np.log2(pa))
            # bayesian            
            self.h_bayes_only = -np.sum(self.p_a_mb*np.log2(self.p_a_mb))

                    
    def updateWeight(self, r):
        if r:
            p_wmc = self.p_a_mb[self.current_action]
            p_rl = self.p_a_mf[self.current_action]
        else:
            p_wmc = 1.0 - self.p_a_mb[self.current_action]
            p_rl = 1.0 - self.p_a_mf[self.current_action]
        self.w[self.current_state] = (p_wmc*self.w[self.current_state])/(p_wmc*self.w[self.current_state] + p_rl * (1.0 - self.w[self.current_state]))
        # self.p_wm[-1].append(self.p_a_mb[self.current_action])
        # self.p_rl[-1].append(self.p_a_mf[self.current_action])
        
    def computeValue(self, s, a, ind):
        self.current_state = s
        self.current_action = a
        self.p = self.uniform[:,:,:]
        self.Hb = self.max_entropy
        self.nb_inferences = 0     
        # print self.Hb, self.parameters['threshold'], self.nb_inferences, self.n_element
        while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:            
            self.inferenceModule()
            self.evaluationModule()                    
        
        self.fusionModule()
        # print ind, self.p_a
        H = -(self.p_a*np.log2(self.p_a)).sum()
        self.N = self.nb_inferences
        # if np.isnan(H): H = 0.005

        self.value[ind] = float(np.log(self.p_a[self.current_action]))
        self.reaction[ind] = float((np.log2(float(self.nb_inferences+1))**self.parameters['sigma'])+H)        

    def chooseAction(self, state):
        self.state[-1].append(state)
        self.current_state = convertStimulus(state)-1
        self.p = self.uniform[:,:,:]
        self.Hb = self.max_entropy
        self.nb_inferences = 0             
        # print self.Hb, self.parameters['threshold'], self.nb_inferences, self.n_element
        while self.Hb > self.parameters['threshold'] and self.nb_inferences < self.n_element:
            self.inferenceModule()
            self.evaluationModule()
        
        self.fusionModule()        
        self.current_action = self.sample(self.p_a)
        self.value.append(float(self.p_a[self.current_action]))
        self.action[-1].append(self.current_action)
        self.weights[-1].append(self.w[self.current_state])
        H = -(self.p_a*np.log2(self.p_a)).sum()
        N = float(self.nb_inferences+1)
        self.Hl = H
        self.reaction[-1].append(float((np.log2(N)**self.parameters['sigma'])+H))        
        self.Hall[-1].append([float(self.Hb), float(self.Hf)])
        self.pdf[-1].append(N)
        return self.actions[self.current_action]

    def updateValue(self, reward):
        r = int((reward==1)*1)
        if not self.sferes:
            self.responses[-1].append(r)        
        # Specific to Collins model
        self.updateWeight(float(r))
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
        r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0        
        # delta = float(r)+self.parameters['gamma']*np.max(self.q_mf[self.current_state])-self.q_mf[self.current_state, self.current_action]                
        self.delta = float(r)-self.q_mf[self.current_state, self.current_action]                        
        self.q_mf[self.current_state, self.current_action] = self.q_mf[self.current_state, self.current_action]+self.parameters['alpha']*self.delta
        # if r>0:        
        #     self.q_mf[self.current_state, self.current_action] = self.q_mf[self.current_state, self.current_action]+self.parameters['alpha']*delta
        # elif r<=0:
        #     self.q_mf[self.current_state, self.current_action] = self.q_mf[self.current_state, self.current_action]+self.parameters['omega']*delta                    




class Keramati():
    """Class that implement Keramati models for action selection
    Use to replicate exp 1 from Keramati & al, 2011
    """
    
    def __init__(self, kalman,depth,phi, rau, sigma, tau):
        self.kalman = kalman
        self.depth = depth
        self.phi = phi
        self.rau = rau
        self.sigma = sigma
        self.tau = tau
        self.gamma = self.kalman.parameters['gamma']
        self.beta = self.kalman.parameters['beta']
        self.actions = kalman.actions; self.states = kalman.states
        self.values = createQValuesDict(kalman.states, kalman.actions)
        self.rfunction = createQValuesDict(kalman.states, kalman.actions)
        self.vpi = dict.fromkeys(self.states,list())
        self.rrate = [0.0]
        self.state = None
        self.action = None
        self.transition = createTransitionDict(['s0','s0','s1','s1'],
                                               ['pl','em','pl','em'],
                                               ['s1','s0','s0','s0'], 's0') #<====VERY BAD==============    NEXT_STATE = TRANSITION[(STATE, ACTION)]        
    def initialize(self):
        self.values = createQValuesDict(self.states, self.actions)
        self.rfunction = createQValuesDict(self.states, self.actions)
        self.vpi = dict.fromkeys(self.states,list())
        self.rrate = [0.0]
        self.state = None
        self.action = None
        self.transition = createTransitionDict(['s0','s0','s1','s1'],
                                               ['pl','em','pl','em'],
                                               ['s1','s0','s0','s0'], 's0')
                
    def chooseAction(self, state):
        self.state = state
        self.kalman.predictionStep()
        n = self.kalman.states.index(self.state)
        t = len(self.actions)*n
        vpi = computeVPIValues(self.kalman.values[n], self.kalman.covariance['cov'].diagonal()[t:t+len(self.actions)])        
        
        for i in range(len(vpi)):
            if vpi[i] >= self.rrate[-1]*self.tau:
                depth = self.depth
                self.values[0][self.values[(self.state, self.actions[i])]] = self.computeGoalValue(self.state, self.actions[i], depth)
            else:
                self.values[0][self.values[(self.state, self.actions[i])]] = self.kalman.values[n, i]

        self.action = getBestActionSoftMax(state, self.values, self.beta)
        return self.action

    def updateValues(self, reward, next_state):
        self.updateRewardRate(reward, delay = 0.0)
        self.kalman.current_state = self.kalman.states.index(self.state)
        self.kalman.current_action = self.kalman.actions.index(self.action)        
        self.kalman.updateValue(reward)
        self.updateRewardFunction(self.state, self.action, reward)
        self.updateTransitionFunction(self.state, self.action)

    def updateRewardRate(self, reward, delay = 0.0):
        self.rrate.append(((1-self.sigma)**(1+delay))*self.rrate[-1]+self.sigma*reward)

    def updateRewardFunction(self, state, action, reward):
        self.rfunction[0][self.rfunction[(state, action)]] = (1-self.rau)*self.rfunction[0][self.rfunction[(state, action)]]+self.rau*reward

    def updateTransitionFunction(self, state, action):
        #This is cheating since the transition is known inside the class
        #Plus assuming the transition are deterministic
        nextstate = self.transition[(state, action)]
        for i in [nextstate]:
            if i == nextstate:
                self.transition[(state, action, nextstate)] = (1-self.phi)*self.transition[(state, action, nextstate)]+self.phi
            else:
                self.transition[(state, action, i)] = (1-self.phi)*self.transition[(state, action, i)]
        
    def computeGoalValue(self, state, action, depth):
        next_state = self.transition[(state, action)]
        tmp = np.max([self.computeGoalValueRecursive(next_state, a, depth-1) for a in xrange(len(self.actions))])
        value =  self.rfunction[0][self.rfunction[(state, action)]] + self.gamma*self.transition[(state, action, next_state)]*tmp
        return value

    def computeGoalValueRecursive(self, state, a, depth):
        action = self.actions[a]
        next_state = self.transition[(state, action)]
        if depth:
            tmp = np.max([self.computeGoalValueRecursive(next_state, a, depth-1) for a in xrange(len(self.actions))])
            return self.rfunction[0][self.rfunction[(state, action)]] + self.gamma*self.transition[(state, action, next_state)]*tmp
        else:
            return self.rfunction[0][self.rfunction[(state, action)]] + self.gamma*self.transition[(state, action, next_state)]*np.max(self.kalman.values[self.kalman.states.index(state)])        
        
