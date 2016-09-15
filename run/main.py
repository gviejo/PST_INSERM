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

parameters = {"alpha":0.7646, "beta":1.3057, "noise":0.0713, "length":4.6741, "gain":0.0, "threshold":3.6129, "gamma":7.5942, "sigma":2.0893, "kappa":0.0079, "shift":0.1079}


model = Sweeping()
fit = model.sferes_call(monkeys['m'], rt_reg_monkeys['m'], parameters)
# print fit[0], fit[1]


