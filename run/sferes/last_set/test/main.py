#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

load and test last set for manuscrit


"""

import sys
import os
from optparse import OptionParser
import numpy as np
from model_g import CSelection_g_2
from model_m import FSelection_m_4
from model_p import CSelection_p_5
from model_s import FSelection_s_2
from model_r2 import FSelection_r_3
from model_all_f5 import FSelection_5
from model_CSelection_4 import CSelection_4
sys.path.append("../../../../src")



#from matplotlib import *
#from pylab import *

#from Sferes import pareto
from itertools import *
from time import sleep
import cPickle as pickle


with open("../../p_test_last_set.pickle", 'rb') as f:
	p_test = pickle.load(f)
# with open("../../p_test2_last_set.pickle", 'rb') as f:
# 	p_test = pickle.load(f)

# LOAD monkeys data
monkeys = {}
N = {}        
rt_reg_monkeys = {}
for s in os.listdir("../../../../data/data_txt_3_repeat/"):
    if "rt_reg.txt" in s:
        rt_reg_monkeys[s.split("_")[0]] = np.genfromtxt("../../../../data/data_txt_3_repeat/"+s)
    else :
        monkeys[s.split(".")[0]] = np.genfromtxt("../../../../data/data_txt_3_repeat/"+s)
        N[s.split(".")[0]] = len(monkeys[s.split(".")[0]])       

data = {}
# on compte le nombre de probleme dans chaque cas
for s in p_test.keys():
	data[s] = {}	
	tmp = []
	for i in [0,4,9,15,22]:
		tmp.append(np.sum(monkeys[s[0]][:,-1] == i))
	tmp = np.array(tmp).astype('float')
	data[s]['proportion'] = tmp/tmp.sum()
 
##########################################
# singe g mixture 2
# singe g fusion 5 rt max
model = CSelection_g_2()
# model = FSelection_5()
# model.test_call(monkeys['g'], rt_reg_monkeys['g'], p_test['g2']['mixture'])
fit = model.sferes_call(monkeys['g'], rt_reg_monkeys['g'], p_test['g2']['mixture'])
data['g2'] = {'rt_model':model.rt_model,
 			'rt_monkey':model.mean_rt}
print "g2", fit+50000, "\n"

##########################################
# singe m fusion 4
# singe m mixture 4 rt max
model = FSelection_m_4()
# model = CSelection_4()
# model.test_call(monkeys['m'], rt_reg_monkeys['m'], p_test['m4']['fusion'])
fit = model.sferes_call(monkeys['m'], rt_reg_monkeys['m'], p_test['m4']['fusion'])
data['m4'] = {'rt_model':model.rt_model,
			'rt_monkey':model.mean_rt}
print 'm4', fit+50000, "\n"
##########################################
# singe p mixture 5
# singe p fusion 5 rt max
model = CSelection_p_5()
# model = FSelection_5()
# model.test_call(monkeys['p'], rt_reg_monkeys['p'], p_test['p5']['mixture'])
fit = model.sferes_call(monkeys['p'], rt_reg_monkeys['p'], p_test['p5']['mixture'])
data['p5'] = {'rt_model':model.rt_model,
			 'rt_monkey':model.mean_rt}
print 'p5', fit+50000, "\n"
##########################################
# singe r mixture 5
# singe r fusion 3 rt max
model = CSelection_p_5()
# model = FSelection_r_3()
# model.test_call(monkeys['r'], rt_reg_monkeys['r'], p_test['r5']['mixture'])
fit = model.sferes_call(monkeys['r'], rt_reg_monkeys['r'], p_test['r5']['mixture'])
data['r5'] = {'rt_model':model.rt_model,
 			'rt_monkey':model.mean_rt}
print 'r5', fit+50000, "\n"
##########################################
# singe s fusion 2
model = FSelection_s_2()
# model = FSelection_5()
# model.test_call(monkeys['r'], rt_reg_monkeys['r'], p_test['r5']['mixture'])
fit = model.sferes_call(monkeys['s'], rt_reg_monkeys['s'], p_test['s2']['fusion'])
data['s2'] = {'rt_model':model.rt_model,
 			'rt_monkey':model.mean_rt}
print 's2', fit+50000, "\n"

with open("/home/viejo/Dropbox/Manuscrit/Chapitre5/monkeys/simulation.pickle", 'wb') as f:
	pickle.dump(data, f)

# with open("/home/viejo/Dropbox/Manuscrit/Chapitre5/monkeys/simulation_rt_max.pickle", 'wb') as f:
# 	pickle.dump(data, f)

# tmp1 = np.zeros((5,30))
# tmp2 = np.zeros((5,30))

# for s,i in zip(monkeys.keys(),np.arange(5)):
# 	for j in xrange(len(monkeys[s])):
# 		if monkeys[s][j,3] >= 1 and monkeys[s][j,2] >= 1:
# 			tmp1[i,monkeys[s][j,-1]] += 1.0
# 		tmp2[i,monkeys[s][j,-1]] += 1.0

