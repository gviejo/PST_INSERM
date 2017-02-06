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

class fusion_1():
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
			if self.sari[i][4]-self.sari[i-1][4] < 0.0 and i > 0:                    
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

	def analysis_call(self, sari, mean_rt, parameters):
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
		## LIST ####
		self.w_list = np.zeros(self.N)
		self.entropy_list = np.zeros((self.N,2))
		self.free_list = np.zeros((self.N,4))
		self.based_list = np.zeros((self.N,4))
		self.biais_list = np.zeros((self.N,4))
		self.delta_list = np.zeros((self.N,4))
		self.inference_list = np.zeros((self.N,1))
		self.Hb_list = np.zeros((self.N, self.parameters['length']+1))
		self.wmean_dict = {i:np.zeros(i+3) for i in xrange(1,6)}
		self.wmean_count = {i:np.zeros(i+3) for i in xrange(1,6)}
		start = 0
		############
		for i in xrange(self.N):                    
			if self.sari[i][4]-self.sari[i-1][4] < 0.0 and i > 0:                    
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
					#######################
					nb_search = np.sum(self.sari[start:i-1,4] == 0)
					nb_repeat = np.sum(self.sari[start:i-1,4] == 1)
					if nb_repeat >= 3 and nb_search <= 5:
						w = (self.max_entropy - self.entropy_list[start:start+nb_search+3,0])/(2.0*self.max_entropy - self.entropy_list[start:start+nb_search+3,0] - self.entropy_list[start:start+nb_search+3,1])
						self.wmean_dict[nb_search] += w
						self.wmean_count[nb_search] += 1.0
					elif nb_repeat < 3 and nb_search <= 5:
						w = (self.max_entropy - self.entropy_list[start:start+nb_search+nb_repeat,0])/(2.0*self.max_entropy - self.entropy_list[start:start+nb_search+nb_repeat,0] - self.entropy_list[start:start+nb_search+nb_repeat,1])
						self.wmean_dict[nb_search][0:nb_search+nb_repeat] += w
						self.wmean_count[nb_search][0:nb_search+nb_repeat] += 1.0
					start = i
					#######################

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
			q_values[0] = self.p_a_mb

			self.fusionModule()
			self.p_ak[0] = self.p_a_final[self.current_action]            
			H = -(self.p_a_final*np.log2(self.p_a_final)).sum()
			reaction[0] = float(H)        
			# reaction[0] = np.log2(0.25)+self.parameters['sigma']*self.Hf
			self.Hb_list[i,0] = self.Hb
			for j in xrange(self.n_element):            
				self.inferenceModule()
				self.evaluationModule()
				self.Hb_list[i,j+1] = self.Hb
				q_values[j+1] = self.p_a_mb
				self.fusionModule()                
				self.p_ak[j+1] = self.p_a_final[self.current_action]                
				H = -(self.p_a_final*np.log2(self.p_a_final)).sum()
				N = self.nb_inferences+1.0
				reaction[j+1] = float(((np.log2(N))**self.parameters['sigma'])+H)
				# reaction[j+1] = self.Hb + self.parameters['sigma']*self.Hf
				# reaction[j+1] = H
				self.sigmoideModule()
				self.p_sigmoide[j+1] = self.pA            
				self.p_decision[j+1] = self.pA*self.p_retrieval[j]            
				self.p_retrieval[j+1] = (1.0-self.pA)*self.p_retrieval[j]                    
				# print j+1, " p_ak=", self.p_ak[j+1], " p_decision=", self.p_decision[j+1], " p_retrieval=", self.p_retrieval[0]
			
			##############            
			self.entropy_list[i,0] = np.dot(self.p_decision, self.Hb_list[i])
			self.entropy_list[i,1] = self.Hf
			self.free_list[i] = self.p_a_mf
			tmp = np.dot(self.p_decision,q_values)            
			self.based_list[i] = tmp/tmp.sum()
			self.biais_list[i] = self.spatial_biases            
			if self.n_element:
				self.inference_list[i] = np.dot(self.p_decision, np.arange(int(self.parameters['length'])+1))
			else:
				self.inference_list[i] = 0            
			##############

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

		##########################
		for k in self.wmean_dict.iterkeys():
			self.wmean_dict[k] = self.wmean_dict[k]/self.wmean_count[k]
		##########################

	def test_call(self, nb_repeat, list_of_problems, parameters):
		self.parameters = parameters
		self.list_of_problems = list_of_problems
		self.N = len(self.list_of_problems)
		self.performance = np.zeros((nb_repeat, self.list_of_problems[:,0].sum(), 3))
		self.timing = np.zeros((nb_repeat, self.list_of_problems[:,0].sum(), 5+3))	


		for k in xrange(nb_repeat):
			self.problem = self.list_of_problems[0,1]
			r = 0.0
			self.max_entropy = -np.log2(1./self.n_action)        		
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
			
			self.p_a_final = np.zeros(self.n_action)
			self.spatial_biases = np.ones(self.n_action) * (1./self.n_action)        

			for i in xrange(self.N):
				if self.list_of_problems[i,0]:
					# START BLOC			
					self.n_element = 0
					self.values_mf = np.zeros(self.n_action)
					self.problem = self.list_of_problems[i,1]
					counter = 0

				# START TRIAL
				self.current_action = self.list_of_problems[i,2]
				r = self.list_of_problems[i,3]

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
				self.reaction[i] = float(np.sum(reaction*np.round(self.p_decision.flatten(),3)))
				self.timing[k,self.list_of_problems[0:i,0].sum()-1,counter] = self.reaction[i]

				self.updateValue(r)			

				counter += 1

				if int(self.list_of_problems[i,3]) == 1:				
					# REPEAT PHASE					
					for j in xrange(3):
						r = 0
						self.p_a_mf = SoftMaxValues(self.values_mf, self.parameters['gamma'])    
						self.Hf = -(self.p_a_mf*np.log2(self.p_a_mf)).sum()
						# BAYESIAN CALL
						self.p = self.uniform[:,:]
						self.Hb = self.max_entropy
						self.nb_inferences = 0
						self.p_a_mb = np.ones(self.n_action)*(1./self.n_action)        	            
						while self.sigmoideModule():  
							self.inferenceModule()
							self.evaluationModule()	                
							self.fusionModule()

						self.current_action = self.sample(self.p_a_final)
						
						if self.current_action == self.problem:
							r = 1
						
						H = -(self.p_a_final*np.log2(self.p_a_final)).sum()
						N = self.nb_inferences+1.0	                
						self.updateValue(r)	     
						self.performance[k,self.list_of_problems[0:i,0].sum()-1,j] = r
						self.timing[k,self.list_of_problems[0:i,0].sum()-1,counter+j] = float(((np.log2(N))**self.parameters['sigma'])+H)                
					


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
		self.p_a_final = SoftMaxValues(self.values_net, float(self.parameters['beta']))

		ninf = np.isinf(self.p_a_final).sum()        
		if ninf:
			print "INF"
		
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
		# Updating model free
		r = (reward==0)*-1.0+(reward==1)*1.0+(reward==-1)*-1.0                               
		self.delta = float(r)-self.values_mf[self.current_action]        
		self.values_mf[self.current_action] = self.values_mf[self.current_action]+self.parameters['alpha']*self.delta                
		# forgetting
		# index = range(self.n_action)
		# index.pop(int(self.current_action))        
		# self.values_mf[index] = self.values_mf[index] + (1.0-self.parameters['kappa']) * (0.0 - self.values_mf[index])    
