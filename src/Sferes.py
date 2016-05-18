#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sferes.py

    class for multi-objective optimization
    to interface with sferes2 : see
    http://sferes2.isir.upmc.fr/
    fitness function is made of Bayesian Information Criterion
    and either Linear Regression
    or possible Reaction Time Likelihood

Copyright (c) 2014 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
import mmap
import numpy as np
from multiprocessing import Pool, Process
from pylab import *
import cPickle as pickle

#from fonctions import *
# from Selection import *
from Models import *
# from HumanLearning import HLearning
#from ColorAssociationTasks import CATS
#from scipy.stats import sem
#from scipy.stats import norm
#from scipy.optimize import leastsq

def unwrap_self_load_data(arg, **kwarg):
    return pareto.loadPooled(*arg, **kwarg)

def unwrap_self_re_test(arg, **kwarg):
    return pareto.poolTest(*arg, **kwarg)



class pareto():
    """
    Explore Pareto Front from Sferes Optimization
    """
    def __init__(self, directory = '', nb_repeat = 3):
        self.directory = directory           

        # LOAD monkeys data
        self.monkeys = {}
        self.N = {}        
        self.rt_reg_monkeys = {}
        for s in os.listdir("../../data/data_txt_"+str(nb_repeat)+"_repeat/"):
            if "rt_reg.txt" in s:
                self.rt_reg_monkeys[s.split("_")[0]] = np.genfromtxt("../../data/data_txt_"+str(nb_repeat)+"_repeat/"+s)
            else :
                self.monkeys[s.split(".")[0]] = np.genfromtxt("../../data/data_txt_"+str(nb_repeat)+"_repeat/"+s)
                self.N[s.split(".")[0]] = len(self.monkeys[s.split(".")[0]])               

        self.models = dict({"fusion":FSelection(),
                            "qlearning":QLearning(),
                            "bayesian":BayesianWorkingMemory(),
                            # "selection":KSelection(),
                            "mixture":CSelection()})

        self.p_order = dict({'fusion':['alpha','beta', 'noise','length', 'gain', 'threshold', 'gamma', 'sigma', 'kappa', 'shift'], 
                            'qlearning':['alpha','beta', 'sigma'],
                            'bayesian':['length','noise','threshold', 'sigma'],
                            'selection':['beta','eta','length','threshold','noise','sigma', 'sigma_rt'],
                            'mixture':['alpha', 'beta', 'noise', 'length', 'weight', 'threshold', 'sigma', 'kappa', 'shift']})

        self.m_order = ['qlearning', 'bayesian', 'selection', 'fusion', 'mixture']
        self.colors_m = dict({'fusion':'r', 'bayesian':'g', 'qlearning':'grey', 'selection':'b', 'mixture':'y'})
        self.markers_type = dict({'distance':'*', 'owa': '^', 'tche': 'o'})
        self.data = dict()
        self.opt = dict()
        self.pareto = dict()
        self.distance = dict()
        self.owa = dict()
        self.tche = dict()
        self.p_test = dict()
        self.mixed = dict()        
        self.indd = dict()
        self.zoom = dict()
        self.timing = dict()
        self.beh = dict()
        if self.directory != '':
            self.simpleLoadData()    
        
        

    def showBrute(self):
        rcParams['ytick.labelsize'] = 8
        rcParams['xtick.labelsize'] = 8        
        fig_brute = figure(figsize = (10,10)) # for each model all subject            
        axes = {}        
        for s,i in zip(self.human.keys(),range(2,16)):
            axes[s] = fig_brute.add_subplot(4,4,i)

        for s in self.human.iterkeys():            
            for m in self.data.iterkeys():
                if s in self.data[m].keys():
                    tmp={n:self.data[m][s][n][self.data[m][s][n][:,0]==np.max(self.data[m][s][n][:,0])] for n in self.data[m][s].iterkeys()}
                    tmp=np.vstack([np.hstack((np.ones((len(tmp[n]),1))*n,tmp[n])) for n in tmp.iterkeys()])
                    ind = tmp[:,3] != 0
                    tmp = tmp[ind]
                    tmp = tmp[tmp[:,3].argsort()][::-1]
                    pareto_frontier = [tmp[0]]
                    for pair in tmp[1:]:
                        if pair[4] >= pareto_frontier[-1][4]:
                            pareto_frontier.append(pair)
                    pareto_frontier = np.array(pareto_frontier)
                    pareto_frontier[:,3] -= (2000+float(len(self.p_order[m]))*np.log(156))                    
                    pareto_frontier[:,4] -= 500
                    axes[s].plot(pareto_frontier[:,3], pareto_frontier[:,4], "-o", color = self.colors_m[m], alpha = 1.0)        
                    axes[s].set_title(s)
                    # axes[s].set_ylim(-10,0.0)

    def loadData(self):
        model_in_folders = os.listdir(self.directory)
        if len(model_in_folders) == 0:
            sys.exit("No model found in directory "+self.directory)

        pool = Pool(len(model_in_folders))
        tmp = pool.map(unwrap_self_load_data, zip([self]*len(model_in_folders), model_in_folders))
        
        for d in tmp:
            self.data[d.keys()[0]] = d[d.keys()[0]]

    def simpleLoadData(self):
        model_in_folders = os.listdir(self.directory)
        if len(model_in_folders) == 0:
            sys.exit("No model found in directory "+self.directory)

        for m in model_in_folders:            
            self.data[m.split("_")[0]] = dict()
            lrun = os.listdir(self.directory+"/"+m)
            order = self.p_order[m.split("_")[0]]
            scale = self.models[m.split("_")[0]].bounds

            for r in lrun:
                print r                
                s = r.split("_")[4]                
                n = int(r.split("_")[5].split(".")[0])                
                if s in self.data[m.split("_")[0]].keys():
                    self.data[m.split("_")[0]][s][n] = np.genfromtxt(self.directory+"/"+m+"/"+r)
                else :
                    self.data[m.split("_")[0]][s] = dict()
                    self.data[m.split("_")[0]][s][n] = np.genfromtxt(self.directory+"/"+m+"/"+r)                                
                for p in order:
                    self.data[m.split("_")[0]][s][n][:,order.index(p)+4] = scale[p][0]+self.data[m.split("_")[0]][s][n][:,order.index(p)+4]*(scale[p][1]-scale[p][0])

    def loadPooled(self, m):         
        data = {m:{}}
        list_file = os.listdir(self.directory+"/"+m)
        order = self.p_order[m]
        scale = self.models[m].bounds
        for r in list_file:
            s = r.split("_")[3]
            n = int(r.split("_")[4].split(".")[0])
            filename = self.directory+"/"+m+"/"+r            
            nb_ind = int(self.tail(filename, 1)[0].split(" ")[1])
            last_gen = np.array(map(lambda x: x[0:-1].split(" "), self.tail(filename, nb_ind+1))).astype('float')
            if s in data[m].keys():
                data[m][s][n] = last_gen
            else:
                data[m][s] = {n:last_gen}
            for p in order:
                data[m][s][n][:,order.index(p)+4] = scale[p][0]+data[m][s][n][:,order.index(p)+4]*(scale[p][1]-scale[p][0])                    
        return data

    def tail(self, filename, n):
        size = os.path.getsize(filename)
        with open(filename, "rb") as f:
            fm = mmap.mmap(f.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ)
            for i in xrange(size-1, -1, -1):
                if fm[i] == '\n':
                    n -= 1
                    if n == -1:
                        break
            return fm[i+1 if i else 0:].splitlines()

    def constructParetoFrontier(self, case = 'r2'):
        # Best log est log(0.25)*nb de problem
        self.best_log = dict()
        for s in self.monkeys.keys():
            self.best_log[s] = np.log(0.25)
            problem = self.monkeys[s][0,4]
            for t in xrange(len(self.monkeys[s])):
                if self.monkeys[s][t,4] != problem:
                    if self.monkeys[s][t,2] - self.monkeys[s][t-1,2] < 0.0:
                        self.best_log[s] += np.log(0.25)

        for m in self.data.iterkeys():
            self.pareto[m] = dict()
            for s in self.data[m].iterkeys():
                worst_log = self.N[s]*np.log(0.25)
                
                self.pareto[m][s] = dict()   
                tmp={n:self.data[m][s][n][self.data[m][s][n][:,0]==np.max(self.data[m][s][n][:,0])] for n in self.data[m][s].iterkeys()}
                tmp=np.vstack([np.hstack((np.ones((len(tmp[n]),1))*n,tmp[n])) for n in tmp.iterkeys()])
                ind = tmp[:,3] != 0
                tmp = tmp[ind]
                tmp = tmp[tmp[:,3].argsort()][::-1]
                pareto_frontier = [tmp[0]]
                for pair in tmp[1:]:
                    if pair[4] >= pareto_frontier[-1][4]:
                        pareto_frontier.append(pair)
                self.pareto[m][s] = np.array(pareto_frontier)

                self.pareto[m][s][:,3] = self.pareto[m][s][:,3] - 50000.0
                self.pareto[m][s][:,4] = self.pareto[m][s][:,4] - 50000.0
                if case == 'r2':
                    self.pareto[m][s][:,3] = 1.0 - (self.pareto[m][s][:,3]/(self.N[s]*np.log(0.25)))
                elif case == 'log':
                    self.pareto[m][s][:,3] = (self.pareto[m][s][:,3]-worst_log)/(self.best_log[s]-worst_log)
                elif case == 'bic':
                    self.pareto[m][s][:,3] = 2*self.pareto[m][s][:,3] - float(len(self.p_order[m]))*np.log(self.N[s])
                    best_bic = 2*self.best_log[s] - float(len(self.p_order[m]))*np.log(self.N[s])
                    worst_bic = 2*worst_log - float(len(self.p_order[m]))*np.log(self.N[s])
                    self.pareto[m][s][:,3] = (self.pareto[m][s][:,3]-worst_bic)/(best_bic-worst_bic)
                elif case == 'aic':
                    self.pareto[m][s][:,3] = 2*self.pareto[m][s][:,3] - 2.0*float(len(self.p_order[m]))
                    best_aic = 2*self.best_log[s] - float(len(self.p_order[m]))*2.0
                    worst_aic = 2*worst_log - float(len(self.p_order[m]))*2.0
                    self.pareto[m][s][:,3] = (self.pareto[m][s][:,3]-worst_aic)/(best_aic - worst_aic)
                self.pareto[m][s][:,4] = 1.0 - ((-self.pareto[m][s][:,4])/(np.power(2.0*self.rt_reg_monkeys[s][:,1], 2).sum()))
                # # on enleve les points negatifs                
                self.pareto[m][s] = self.pareto[m][s][(self.pareto[m][s][:,3:5]>0).prod(1)==1]

    def constructMixedParetoFrontier(self):
        # subjects = set.intersection(*map(set, [self.pareto[m].keys() for m in self.pareto.keys()]))
        subjects = self.pareto['fusion'].keys()
        for s in subjects:            
            tmp = []            
            for m in self.pareto.iterkeys():
                if s in self.pareto[m].keys():
                    tmp.append(np.hstack((np.ones((len(self.pareto[m][s]),1))*self.m_order.index(m), self.pareto[m][s][:,0:5])))            
            tmp = np.vstack(tmp)            
            tmp = tmp[tmp[:,4].argsort()][::-1]                        
            if len(tmp):
                self.mixed[s] = []
                self.mixed[s] = [tmp[0]]
                for pair in tmp[1:]:
                    if pair[5] >= self.mixed[s][-1][5]:
                        self.mixed[s].append(pair)
                self.mixed[s] = np.array(self.mixed[s])            

    def removeIndivDoublons(self):
        for m in self.pareto.iterkeys():
            for s in self.pareto[m].iterkeys():
                if len(self.pareto[m][s]):
                    # start at column 5; for each parameters columns, find the minimal number of value
                    # then mix all parameters
                    tmp = np.zeros((len(self.pareto[m][s]),len(self.p_order[m])))
                    for i in xrange(len(self.p_order[m])):
                        tmp[:,i][np.unique(self.pareto[m][s][:,i+5], return_index = True)[1]] = 1.0
                    self.pareto[m][s] = self.pareto[m][s][tmp.sum(1)>0]

    def rankDistance(self):
        self.p_test['distance'] = dict()        
        self.indd['distance'] = dict()
        for s in self.mixed.iterkeys():
            self.distance[s] = np.zeros((len(self.mixed[s]), 3))
            self.distance[s][:,1] = np.sqrt(np.sum(np.power(self.mixed[s][:,4:6]-np.ones(2), 2),1))
            ind_best_point = np.argmin(self.distance[s][:,1])
            best_point = self.mixed[s][ind_best_point,4:6]
            self.distance[s][:,0] = np.sqrt(np.sum(np.power(self.mixed[s][:,4:6]-best_point,2),1))
            self.distance[s][0:ind_best_point,0] = -1.0*self.distance[s][0:ind_best_point,0]
            self.distance[s][0:ind_best_point,2] = np.arange(-ind_best_point,0)
            self.distance[s][ind_best_point:,2] = np.arange(0, len(self.distance[s])-ind_best_point)
            # Saving best individual                        
            best_ind = self.mixed[s][ind_best_point]
            self.indd['distance'][s] = best_ind
            m = self.m_order[int(best_ind[0])]            
            tmp = self.pareto[m][s][(self.pareto[m][s][:,0] == best_ind[1])*(self.pareto[m][s][:,2] == best_ind[3])]
            assert len(tmp) == 1
            self.p_test['distance'][s] = dict({m:dict(zip(self.p_order[m],tmp[0,5:]))})

    def rankOWA(self):
        self.p_test['owa'] = dict()
        self.indd['owa'] = dict()
        for s in self.mixed.iterkeys():
            tmp = self.mixed[s][:,4:6]
            value = np.sum(np.sort(tmp)*[0.5, 0.5], 1)
            self.owa[s] = value
            ind_best_point = np.argmax(value)
            # Saving best indivudual
            best_ind = self.mixed[s][ind_best_point]
            self.indd['owa'][s] = best_ind
            m = self.m_order[int(best_ind[0])]
            tmp = self.pareto[m][s][(self.pareto[m][s][:,0] == best_ind[1])*(self.pareto[m][s][:,2] == best_ind[3])]
            assert len(tmp) == 1
            self.p_test['owa'][s] = dict({m:dict(zip(self.p_order[m],tmp[0,5:]))})            

    def rankTchebytchev(self, lambdaa = 0.5, epsilon = 0.001):
        self.p_test['tche'] = dict()
        self.indd['tche'] = dict()
        for s in self.mixed.iterkeys():
            tmp = self.mixed[s][:,4:6]
            ideal = np.max(tmp, 0)
            nadir = np.min(tmp, 0)
            value = lambdaa*((ideal-tmp)/(ideal-nadir))
            value = np.max(value, 1)+epsilon*np.sum(value,1)
            self.tche[s] = value
            ind_best_point = np.argmin(value)
            # Saving best individual
            best_ind = self.mixed[s][ind_best_point]
            self.indd['tche'][s] = best_ind
            m = self.m_order[int(best_ind[0])]
            tmp = self.pareto[m][s][(self.pareto[m][s][:,0] == best_ind[1])*(self.pareto[m][s][:,2] == best_ind[3])]
            assert len(tmp) == 1
            self.p_test['tche'][s] = dict({m:dict(zip(self.p_order[m],tmp[0,5:]))})                        

    def retrieveRanking(self):
        xmin = 0.0
        ymin = 0.0
        for s in self.mixed.iterkeys():
            self.zoom[s] = np.hstack((self.mixed[s][:,4:6], self.distance[s][:,1:2], np.vstack(self.owa[s]), np.vstack(self.tche[s]), np.vstack(self.mixed[s][:,0])))
 
    def evaluate(self):
        for o in self.p_test.keys():
            self.beh[o] = dict()
            for s in self.p_test[o].keys():
                self.beh[o][s] = dict()
                m = self.p_test[o][s].keys()[0]
                parameters = self.p_test[o][s][m]
                model = self.models[m]
                fit = model.sferes_call(self.monkeys[s], self.rt_reg_monkeys[s], parameters)
                data = self.data[m][s][int(self.indd[o][s][1])]
                line = np.where((data[:,0] == self.indd[o][s][2]) & ( data[:,1] == self.indd[o][s][3]))                
                print o, s, m
                print fit[0], fit[1]
                print data[line,2][0,0] - 50000.0, data[line,3][0,0] - 50000.0, "\n"
                self.beh[o][s][m] = model.rt_model

    def writeParameters(self, filename_):
        "Nicely print parameters in a file"
        filename = filename_+"_best_parameters.txt"
        with open(filename, 'w') as f:
            for o in self.p_test.keys():
                f.write(o+"\n")
                for s in self.p_test[o].iterkeys():
                    # f.write(m+"\n")                
                    f.write(s+"\n")
                    # for s in subjects:
                    m = self.p_test[o][s].keys()[0]
                    line=m+"\t"+" \t".join([k+"="+str(np.round(self.p_test[o][s][m][k],4)) for k in self.p_order[m]])+"\tloglikelihood = "+str(self.indd[o][s][4])+"\n"      
                    f.write(line)                
                    f.write("\n")

    def writePlot(self, name):
        rcParams['ytick.labelsize'] = 8
        rcParams['xtick.labelsize'] = 8        
        
        fig_1 = figure(figsize = (10,5)) # for each model all subject            
        i = 1
        subjects = self.pareto['fusion'].keys()        
        m_order = []
        for s in subjects:
            ax = fig_1.add_subplot(2,3,i)
            for m in self.pareto.keys():
                m_order.append(m)
                if s in self.pareto[m].keys():
                    ax.plot(self.pareto[m][s][:,3], self.pareto[m][s][:,4], 'o-', color = self.colors_m[m])
            ax.set_title(s)
            ax.grid()
            if i == 1 or i == 4:
                ax.set_ylabel("fit to RT")
            if i == 3 or i == 4 or i == 5:
                ax.set_xlabel("fit to choice")
            
            i+=1

        line2 = tuple([Line2D(range(1),range(1),marker='o', markersize = 2, markeredgecolor = self.colors_m[m], alpha=1.0,color=self.colors_m[m], linewidth = 2) for m in self.pareto.keys()])
        figlegend(line2,tuple(m_order), loc = 'lower right', bbox_to_anchor = (0.92, 0.18), fontsize = 10)

        fig_1.savefig(name+"_pareto_front.pdf")
                
        fig_2 = figure(figsize = (10, 5)) 
        i = 1
        subjects = self.mixed.keys()
        for s in subjects:
            ax = fig_2.add_subplot(2,3,i)
            ax.plot(self.mixed[s][:,4], self.mixed[s][:,5], '-', color = 'grey')
            for m in np.unique(self.mixed[s][:,0]):
                ind = self.mixed[s][:,0] == m
                ax.plot(self.mixed[s][ind,4], self.mixed[s][ind,5], 'o', color = self.colors_m[self.m_order[int(m)]])
            ax.plot(self.zoom[s][np.argmin(self.zoom[s][:,2]),0], self.zoom[s][np.argmin(self.zoom[s][:,2]),1], '*', markersize = 10)
            ax.plot(self.zoom[s][np.argmax(self.zoom[s][:,3]),0], self.zoom[s][np.argmax(self.zoom[s][:,3]),1], '^', markersize = 10)
            ax.plot(self.zoom[s][np.argmin(self.zoom[s][:,4]),0], self.zoom[s][np.argmin(self.zoom[s][:,4]),1], 'o', markersize = 10)            
            ax.set_title(s)    
            ax.grid()
            if i == 1 or i == 4:
                ax.set_ylabel("fit to RT")
            if i == 3 or i == 4 or i == 5:
                ax.set_xlabel("fit to choice")
            i+=1 

        line2 = tuple([Line2D(range(1),range(1),marker='o', markersize = 2, markeredgecolor = self.colors_m[m], alpha=1.0,color=self.colors_m[m], linewidth = 2) for m in self.pareto.keys()])
        figlegend(line2,tuple(m_order), loc = 'lower right', bbox_to_anchor = (0.92, 0.18), fontsize = 10)

        fig_2.savefig(name+"_mixed_pareto_front.pdf")

        fig_3 = figure(figsize = (10, 5))
        subplots_adjust(wspace = 0.3, left = 0.1, right = 0.9)

        i = 1        
        for s in subjects:
            for n in xrange(1, 6):            
                ax = fig_3.add_subplot(5,5,i)
                for o in self.beh.keys():                    
                    m = self.beh[o][s].keys()[0]
                    index = np.where(self.rt_reg_monkeys[s][:,0] == n)[0]
                    ax.plot(self.beh[o][s][m][index], self.markers_type[o]+'-', color = self.colors_m[m], label = o, alpha = 0.6)

                ax.plot(self.rt_reg_monkeys[s][index,1], '-', color = 'black')
                ax.fill_between(np.arange(len(index)), self.rt_reg_monkeys[s][index,1]+self.rt_reg_monkeys[s][index,2], self.rt_reg_monkeys[s][index,1]-self.rt_reg_monkeys[s][index,2], color = 'black', alpha = 0.4)                
                ax.set_xlim(-1, len(index))
                ax.axvline(self.rt_reg_monkeys[s][index[0],0]-0.5,  color = 'grey', alpha = 0.5)
                if i < 21:
                    ax.set_xticks([])
                if i in [1,6,11,16,21]:
                    ax.set_ylabel(s)
                if i >= 21:
                    # ax.set_xticks(np.arange(1,len(index)+1, 2))
                    ax.set_xlabel(str(int(self.rt_reg_monkeys[s][index[0],0]))+" search | 7 repeat", fontsize = 7)

                i+=1

        fig_3.savefig(name+"_evaluation_sferes_call.pdf")

     

























    def rankIndividualStrategy(self):
        # order is distance, owa , tchenbytchev
        data = {}
        p_test = {}
        self.best_ind_single_strategy = dict()
        for m in ['bayesian', 'qlearning']:
            p_test[m] = dict({'tche':dict(),'owa':dict(),'distance':dict()})
            data[m] = dict()
            subjects = self.pareto[m].keys()
            self.best_ind_single_strategy[m] = dict()
            for s in subjects:
                if len(self.pareto[m][s]):                
                    data[m][s] = np.zeros((self.pareto[m][s].shape[0],5))                
                    # pareto position
                    data[m][s][:,0:2] = self.pareto[m][s][:,3:5]
                    # tchenbytchev ranking
                    lambdaa = 0.5
                    epsilon = 0.001
                    tmp = self.pareto[m][s][:,3:5]
                    ideal = np.max(tmp, 0)                
                    nadir = np.min(tmp, 0)
                    value = lambdaa*((ideal-tmp)/(ideal-nadir))
                    value = np.max(value, 1)+epsilon*np.sum(value,1)
                    data[m][s][:,4] = value
                    ind_best_point = np.argmin(value)            
                    best_ind = self.pareto[m][s][ind_best_point]
                    self.best_ind_single_strategy[m][s] = best_ind
                    p_test[m]['tche'][s] = dict({m:dict(zip(self.p_order[m],best_ind[5:]))})                                          
                    # owa ranking
                    data[m][s][:,3] = np.sum(np.sort(tmp)*[0.5, 0.5], 1)                    
                    ind_best_point = np.argmax(data[m][s][:,3])            
                    best_ind = self.pareto[m][s][ind_best_point]
                    p_test[m]['owa'][s] = dict({m:dict(zip(self.p_order[m],best_ind[5:]))})
                    # distance ranking
                    data[m][s][:,2] = np.sqrt(np.sum(np.power(tmp-np.ones(2), 2),1))
                    ind_best_point = np.argmin(data[m][s][:,2])
                    best_ind = self.pareto[m][s][ind_best_point]
                    p_test[m]['distance'][s] = dict({m:dict(zip(self.p_order[m],best_ind[5:]))})
        return data, p_test
            
    def classifySubject(self):
        models = self.data.keys()                
        subjects = self.human.keys()
        self.values = dict()
        self.extremum = dict()                
        for s in subjects:
            self.extremum[s] = dict()            
            self.values[s] = dict()
            data_best_ind = dict()
            for m in models:
                self.extremum[s][m] = dict()
                self.values[s][m] = dict()
                data = []
                for i in self.data[m][s].iterkeys():
                    #max_gen = np.max(self.data[m][s][i][:,0])
                    #size_max_gen = np.sum(self.data[m][s][i][:,0]==max_gen)
                    tmp = np.hstack((np.ones((len(self.data[m][s][i]),1))*i,self.data[m][s][i]))                    
                    #tmp = np.hstack((np.ones((size_max_gen,1))*i,self.data[m][s][i][-size_max_gen:]))
                    data.append(tmp)
                data = np.vstack(data)
                data[:,3]-=2000.0
                # LOG                
                self.values[s][m]['log'] = np.max(data[:,3])
                best_ind = np.argmax(data[:,3])
                data_best_ind[m] = data[best_ind,5:]
                gen = data[best_ind,1]
                ind = data[best_ind,2]
                # print s, data[best_ind,0], gen, ind
                self.extremum[s][m] = dict(zip(self.p_order[m][0:],data_best_ind[m]))                
                # BIC
                self.values[s][m]['bic'] = 2.0*self.values[s][m]['log'] - float(len(self.p_order[m])-1)*np.log(self.N)
               
        self.best_extremum = dict({'bic':{m:[] for m in models},'log':{m:[] for m in models}})
        self.p_test_extremum = dict({'bic':{},'log':{}})
        for s in self.values.iterkeys():
            for o in ['log', 'bic']:
                best = np.argmax([self.values[s][m][o] for m in models])
                self.best_extremum[o][models[best]].append(s)
                self.p_test_extremum[o][s] = {models[best]:self.extremum[s][models[best]]}

        # #writing parameters because fuck you that's why
        # with open("parameters.txt", 'w') as f:
        #     # for m in models:
        #     for s in subjects:
        #         # f.write(m+"\n")                
        #         f.write(s+"\n")
        #         # for s in subjects:
        #         for m in models:                    
        #             line=m+"\t"+" \t".join([k+"="+str(np.round(self.extremum[s][m][k],4)) for k in self.p_order[m][0:-1]])+"\tloglikelihood = "+str(self.values[s][m]['log'])+"\n"      
        #             f.write(line)                
        #         f.write("\n")


        return 

    def timeConversion(self):
        for o in self.p_test.iterkeys():
            self.timing[o] = dict()
            for s in self.p_test[o].iterkeys():
                self.timing[o][s] = dict()
                m = self.p_test[o][s].keys()[0]
                parameters = self.p_test[o][s][m]
                # MAking a sferes call to compute a time conversion
                with open(self.case+"/"+s+".pickle", "rb") as f:
                    data = pickle.load(f)
                self.models[m].__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes = True)
                opt = EA(data, s, self.models[m])                                
                for i in xrange(opt.n_blocs):
                    opt.model.startBloc()
                    for j in xrange(opt.n_trials):
                        opt.model.computeValue(opt.state[i,j]-1, opt.action[i,j]-1, (i,j))
                        opt.model.updateValue(opt.responses[i,j])
                opt.fit[0] = float(np.sum(opt.model.value)) + 2000.0
                self.timing[o][s][m] = [np.median(opt.model.reaction)]
                opt.model.reaction = opt.model.reaction - np.median(opt.model.reaction)
                self.timing[o][s][m].append(np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25))
                opt.model.reaction = opt.model.reaction / (np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25))        
                self.timing[o][s][m] = np.array(self.timing[o][s][m])
                opt.fit[1] = float(-opt.leastSquares()) + 500.0
                # # Check if coherent                 
                ind = self.indd[o][s]
                m2 = self.m_order[int(ind[0])]
                if m != m2:
                    sys.exit()
                real = self.data[m][s][int(ind[1])][ind[3]][2:4]
                
                if np.sum(np.round(real,2)==np.round(opt.fit, 2))!= 2:
                    print o, s, m
                    print "from test :", opt.fit
                    print "from sferes :", real

    def timeConversion_singleStrategy(self, p_test, rank):
        o = 'tche'
        timing = {o:{}}
        for m in ['bayesian','qlearning']:
            timing[o][m] = dict()
            for s in p_test[m][o].iterkeys():
                timing[o][m][s] = dict()
                parameters = p_test[m][o][s][m]
                # MAking a sferes call to compute a time conversion
                with open(self.case+"/"+s+".pickle", "rb") as f:
                    data = pickle.load(f)
                self.models[m].__init__(['s1', 's2', 's3'], ['thumb', 'fore', 'midd', 'ring', 'little'], parameters, sferes = True)
                opt = EA(data, s, self.models[m])                                
                for i in xrange(opt.n_blocs):
                    opt.model.startBloc()
                    for j in xrange(opt.n_trials):
                        opt.model.computeValue(opt.state[i,j]-1, opt.action[i,j]-1, (i,j))
                        opt.model.updateValue(opt.responses[i,j])                
                # opt.fit[0] = float(np.sum(opt.model.value))
                opt.fit[0] = 1.0 - (float(np.sum(opt.model.value)))/(self.N*np.log(0.2))
                timing[o][m][s][m] = [np.median(opt.model.reaction)]
                opt.model.reaction = opt.model.reaction - np.median(opt.model.reaction)
                timing[o][m][s][m].append(np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25))
                opt.model.reaction = opt.model.reaction / (np.percentile(opt.model.reaction, 75)-np.percentile(opt.model.reaction, 25))        
                timing[o][m][s][m] = np.array(timing[o][m][s][m])
                opt.fit[1] = float(-opt.leastSquares())
                #opt.fit[1] = 1.0 - (-opt.fit[1])/(np.power(2*self.human[s]['mean'][0], 2).sum())
                # Check if coherent                                 
                real = self.best_ind_single_strategy[m][s][3:5]
                
                if np.sum(np.round(real,2)==np.round(opt.fit, 2))!= 2:
                    print o, s, m
                    print "from test :", opt.fit
                    print "from sferes :", real
        return timing
