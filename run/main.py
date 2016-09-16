#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To hand search a solution for m and p monkeys

import sys
import os
import numpy as np
from pylab import *
sys.path.append("../src")

from Models import *


monkeys = {}
N = {}        
rt_reg_monkeys = {}
for s in os.listdir("../data/data_txt_3_repeat/"):
	if "rt_reg.txt" in s:
		rt_reg_monkeys[s.split("_")[0]] = np.genfromtxt("../data/data_txt_3_repeat/"+s)
	else :
		monkeys[s.split(".")[0]] = np.genfromtxt("../data/data_txt_3_repeat/"+s)
		N[s.split(".")[0]] = len(monkeys[s.split(".")[0]])               

# 

parameters = {'alpha': 0.000173561,
 'beta': 57.088200000000001,
 'gain': 8371.4200016285813,
 'gamma': 70.217200000000005,
 'kappa': 0.97097299999999997,
 'length': 2.9978920000000002,
 'noise': 0.044231400000000004,
 'shift': 0.0,
 'sigma': 20.0,
 'threshold': 14.2363}


model = Sweeping()
fit = model.sferes_call(monkeys['p'], rt_reg_monkeys['p'], parameters)
print fit[0], fit[1]




