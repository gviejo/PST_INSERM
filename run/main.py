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

parameters = {"alpha":0.9199, "beta":2.0662, "noise":0.1, "length":2.9598, "gain":4.1754, "threshold":4.7774, "gamma":1.1308,
 "sigma":0.3297, "kappa":0.0, "shift":0.0404}

model = Sweeping()
fit = model.sferes_call(monkeys['g'], rt_reg_monkeys['g'], parameters)
print fit[0], fit[1]


