#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cPickle as pickle
import os

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




######################################################################################################
# PLOT ###############################################################################################
######################################################################################################
import matplotlib as mpl
mpl.use('pgf')
def figsize(scale):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 8,               # LaTeX default is 10pt font.
    "font.size": 8,
    "legend.fontsize": 10,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.5),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *



figure(figsize = figsize(1))
x = 1.0
wid = 0.17
ax = subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
for s in new.keys():
	xpos = np.arange(x, x+1.0, wid)[0:5]	
	new[s].append((float(monkeys[s])*np.log(0.25))*-1.0)
	colors = list(np.array(colors_m.values())[np.argsort(new[s])])	
	
	rect = (bar(left = xpos, height = np.sort(new[s]), width = wid, color = colors ))
	

	x+=1.0



tmp = legend_m.values()
legend((rect[0],rect[2], rect[4], rect[1], rect[3]), (tmp[2],tmp[3],tmp[4],tmp[1],tmp[0]), frameon = False, loc = 'upper center', bbox_to_anchor = (0.5, 1.2), fancybox = False, shadow = False, ncol = 2)
# line2 = tuple([plt.Patch(range(1),range(1),alpha=1.0,color=colors_m[m], linewidth = 15) for m in ['qlearning', 'bayesian', 'mixture', 'fusion', 'random']])
# plt.figlegend(line2,tuple([legend_m[m] for m in ['qlearning', 'bayesian', 'mixture', 'fusion', 'random']]) , loc = 'lower right', bbox_to_anchor = (0.48, 0.74))

locator_params(nbins = 5)

xlabel("Monkey")
ylabel("-BIC")
xticks(np.arange(1.4, 6.5), new.keys())
xlim(0.9, 5.9)
yticks([0, 10000,20000,30000,40000])

savefig("fig3_bic.pdf", dpi = 900, facecolor = 'white', bbox_inches = 'tight')
os.system("evince fig3_bic.pdf &")