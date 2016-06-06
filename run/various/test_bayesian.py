#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import os
from optparse import OptionParser
import numpy as np

sys.path.append("../../src")

from Models import BayesianWorkingMemory

from matplotlib import *
from pylab import *

# MONKEYS DATA
monkeys = {}
N = {}        
rt_reg_monkeys = {}
for s in os.listdir("../../data/data_txt_3_repeat/"):
    if "rt_reg.txt" in s:
        rt_reg_monkeys[s.split("_")[0]] = np.genfromtxt("../../data/data_txt_3_repeat/"+s)
    else :
        monkeys[s.split(".")[0]] = np.genfromtxt("../../data/data_txt_3_repeat/"+s)
        N[s.split(".")[0]] = len(monkeys[s.split(".")[0]])               



parameters = "length=4.3223, noise=0.0015, threshold=0.01, sigma=0.0"
parameters = {p.split("=")[0]:float(p.split("=")[1]) for p in parameters.split(", ")}
parameters['length'] = int(parameters['length'])
model = BayesianWorkingMemory()

s = 's'
model.analysis_call(monkeys[s], rt_reg_monkeys[s], parameters)



#####################
t_start = 0
t_stop = 30
figure()
subplot(411)
plot(model.sari[t_start:t_stop,-1], '--')
plot(model.sari[t_start:t_stop,2]*0.25, 'o-')
ylabel("action")

subplot(412)
plot(model.entropy_list[t_start:t_stop], 'o-')
plot(model.sari[t_start:t_stop,-1], '--')
ylabel("entropy")

subplot(413)
plot(model.inference_list[t_start:t_stop], 'o-')
plot(model.sari[t_start:t_stop,-1], '--')
ylabel("N")

subplot(414)
[plot(model.free_list[t_start:t_stop,i], 'o-', label = str(i)) for i in xrange(4)]
plot(model.sari[t_start:t_stop,-1], '--')
ylabel("Q(free)")
legend()

show()


show()
sys.exit()

#####################
w_evolution = np.zeros((2,len(model.rt_model)))
entropy_evolution = np.zeros((len(model.rt_model), 2))
for i in xrange(w_evolution.shape[1]):
	w_evolution[0,i] = np.mean(model.w_list[model.sari[:,3] == i])
	w_evolution[1,i] = np.std(model.w_list[model.sari[:,3] == i])
	entropy_evolution[i,0] = np.mean(model.entropy_list[:,0][model.sari[:,3] == i])
	entropy_evolution[i,1] = np.mean(model.entropy_list[:,1][model.sari[:,3] == i])


fig_3 = figure(figsize = (10, 5))
subplots_adjust(wspace = 0.3, left = 0.1, right = 0.9)

i = 1        

for n in xrange(1, 6):            
    ax = fig_3.add_subplot(2,3,i)
    index = np.where(rt_reg_monkeys[s][:,0] == n)[0]
    
    ax.errorbar(range(len(index)), w_evolution[0,index], w_evolution[1,index])
    ax.axvline(rt_reg_monkeys[s][index[0],0]-0.5,  color = 'grey', alpha = 0.5)
    ax.plot(entropy_evolution[index,0],'o-', label = 'Hb')
    ax.plot(entropy_evolution[index,1],'*-', label = 'Hf')
    ax.legend()
    # ax.set_ylim(0,1)
    if i < 21:
        ax.set_xticks([])
    if i in [1,6,11,16,21]:
        ax.set_ylabel(s)
    if i >= 21:
        ax.set_xlabel(str(int(rt_reg_monkeys[s][index[0],0]))+" search | 7 repeat", fontsize = 7)


    i+=1

show()