#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pylab import *
import cPickle as pickle

monkeys =   { 'g':12701,
              'm':34752,
              'p':27692,
              'r':11634,
              's':13348 }

models_to_id = dict({	'fusion':1,
						'mixture':2,
						'bayesian':3,
						'qlearning':4})

colors_m = dict({'fusion':'#F1433F',
                'bayesian':'#D5A253',
                'qlearning': '#6E8243',
                # 'selection':'#70B7BA',
                'mixture':'#3D4C53',
                'random':'white'})

legend_m = dict({'fusion':'Entropy-based coordination',
                'bayesian':'Bayesian Working Memory',
                'qlearning':'Q-Learning',
                # 'selection': 'VPI-based selection',
                'mixture': 'Weight-based mixture',
                'random': 'Random choice'})


with open("p_test_v1.pickle", 'rb') as f:
	data = pickle.load(f)

new = {}
m_order = models_to_id.keys()
for s in data.keys():
	new[s] = []
	tmp = data[s]['best_choice'].keys()
	for m in models_to_id.keys():
		for i in tmp:
			if m == i.split("_")[0]:
				new[s].append(float(i.split("_")[1])*-1.0)



figure()
x = 1.0
wid = 0.17

for s in new.keys():
	xpos = np.arange(x, x+1.0, wid)[0:5]	
	new[s].append((float(monkeys[s])*np.log(0.25))*-1.0)
	colors = list(np.array(colors_m.values())[np.argsort(new[s])])	
	
	rect = (bar(left = xpos, height = np.sort(new[s]), width = wid, color = colors ))
	

	x+=1.0
tmp = legend_m.values()
legend((rect[0],rect[2], rect[4], rect[1], rect[3]), (tmp[2],tmp[3],tmp[4],tmp[1],tmp[0]), loc = 'upper center', bbox_to_anchor = (0.5, 1.25), fancybox = True, shadow = True, ncol = 2)
# line2 = tuple([plt.Patch(range(1),range(1),alpha=1.0,color=colors_m[m], linewidth = 15) for m in ['qlearning', 'bayesian', 'mixture', 'fusion', 'random']])
# plt.figlegend(line2,tuple([legend_m[m] for m in ['qlearning', 'bayesian', 'mixture', 'fusion', 'random']]) , loc = 'lower right', bbox_to_anchor = (0.48, 0.74))

locator_params(nbins = 5)

xlabel("Monkey")
ylabel("-BIC")
xticks(np.arange(1.4, 6.5), new.keys())
xlim(0.9, 5.9)
yticks([0, 10000,20000,30000,40000])

savefig("fig1_bic.pdf", dpi = 900, facecolor = 'white', bbox_inches = 'tight')