#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import numpy as np
import cPickle as pickle
import sys, os

with open("pareto2.pickle", 'rb') as handle:
    data = pickle.load(handle)

with open("position.pickle", 'rb') as handle:
	pos = pickle.load(handle)

with open("to_compare_value.pickle", 'rb') as handle:
	value = pickle.load(handle)

tmp = {}
for k in pos.iterkeys():
	tmp[k[0]] = pos[k]
pos = tmp

##################################
# PLOT PARAMETERS ################
##################################
import matplotlib as mpl
# mpl.use('pgf')
def figsize(scale):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean*2.              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 5,               # LaTeX default is 10pt font.
    "font.size": 5,
    "legend.fontsize": 6,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 4,
    "ytick.labelsize": 4,
    "figure.figsize": figsize(1),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ],
    "lines.markeredgewidth" : 0.0,
    "axes.linewidth"      	: 0.5,
    "ytick.major.size"		: 1.5,
    "xtick.major.size"		: 1.5
    }    
mpl.rcParams.update(pgf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *

figure(figsize = figsize(1.0))
subplots_adjust(hspace = 0.7, wspace = 0.4)
outer = gridspec.GridSpec(5,1)
dpi = 900
alpha = 0.8
msize = 1.7
lwidth = 1.0
elwidth = 0.4
cpsize = 1.1
dashes = ['-', '--', ':']
colors = ['blue','red','green']
colors_m = dict({'fusion':'#F1433F',
                'bayesian':'#D5A253',
                'qlearning': '#6E8243',
                # 'selection':'#70B7BA',
                'mixture':'#3D4C53'})
legend_m = dict({'fusion':r'$Coordination\ par\ Entropie$',
                'bayesian':r'$M\acute{e}moire\ de\ travail\ bay\acute{e}sienne$',
                'qlearning':r'$Q-Learning$',
                # 'selection': r'$S\acute{e}lection\ par\ VPI$',
                'mixture': r'$M\acute{e}lange\ pond\acute{e}r\acute{e}$'})
markers = ['^', 'o', 'p', 's', '*']
xlimit = dict({0:(0.45,0.65),
			   1:(0.45,0.65),
			   2:(0.4,0.6),
			   3:(0.6,0.70),
			   4:(0.6,0.75)})
ylimit = dict({0:(0.4,0.7),
			   1:(0.2,0.7),
			   2:(0.2,0.5),
			   3:(0.64,0.70),
			   4:(0.95,1.0)})

n_subjects = len(data.keys())
subjects= data.keys()

for i in xrange(n_subjects):
	gs = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec = outer[i], wspace = 0.32)
	# PARETO
	ax1 = subplot(gs[0])
	ax1.get_xaxis().tick_bottom()
	ax1.get_yaxis().tick_left()
	s = subjects[i]
	for m in data[s].keys():			
		positif = (data[s][m][:,4]>0)*(data[s][m][:,5]>0)		
		# les sets dispo dans le front de pareto
		# sets = np.unique(data[s][m][positif,0])
		ax1.plot(data[s][m][positif,4], data[s][m][positif,5], '-o', color = colors_m[m], linewidth = lwidth, markersize = msize, markeredgewidth = 0.0)
		# for j in sets:
		# 	index = (data[s][m][:,0] == j)*positif
		# 	ax1.plot(data[s][m][index,4], data[s][m][index,5] , markers[int(j-1)], markerfacecolor = 'white',  markeredgewidth = 1.0, markersize = msize, markeredgecolor = colors_m[m])			
			
	ax1.plot(pos[s][0], pos[s][1], '*', markersize = 2*msize, color = 'black')
	ax1.locator_params(nbins=5)	

	if i == 4:
		ax1.set_xlim(0.6,0.75)
		ax1.set_ylim(0.85,1.0)
	else:
		ax1.set_xlim(0,1)
		ax1.set_ylim(0,1)
	
	ax1.set_xlabel("Fit to choice")
	ax1.set_ylabel("Fit to RT")
	# ax1.set_title("Monkey "+subjects[i], y = 0.98)

	# BAR
	ax2 = subplot(gs[1])
	ax2.spines['right'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax2.yaxis.set_ticks_position('left')
	ax2.xaxis.set_ticks_position('bottom')	
	x = np.arange(1, 8)	
	for m,k in zip(['mixture', 'fusion'], range(2)):
		positif = (data[s][m][:,4]>0)*(data[s][m][:,5]>0)					
		y = np.zeros(7)	
		for j in np.arange(1, 8):			
			y[int(j-1)] = np.sum(data[s][m][positif,0] == j)
		y = y/y.sum()			
		ax2.bar(x, y, 0.3, color = colors_m[m], linewidth = 0)
		x = x + 0.3
	
	ax2.set_xticks(np.arange(1,8)+0.3)
	ax2.set_xticklabels(np.arange(1,8).astype('str'))
	ax2.set_xlabel("Variation")
	ax2.set_xlim(0.9, 8)
	ax2.locator_params(axis='y',nbins=4)	
	ax2.set_ylabel("\%")
	
	ax2.set_title("Monkey "+subjects[i], y = 0.98)

	# VALUE
	ax3 = subplot(gs[2])	
	ax3.spines['top'].set_visible(False)
	ax3.yaxis.set_ticks_position('left')
	ax3.xaxis.set_ticks_position('bottom')		
	# value[s]['test'] = value[s]['test'] - value[s]['tche'].min()-value[s]['test'].min()
	index = value[s]['test'] != value[s]['test'][0]
	x = np.arange(len(value[s]['test']))
	ax3.plot(x, value[s]['tche'], '-', linewidth = lwidth*2, alpha = 0.5, color = 'black')	
	ax4 = ax3.twinx()	
	ax4.plot(x[index], value[s]['test'][index], '-', linewidth = 0.5, alpha = 1, color = 'black')			
	ax4.locator_params(axis = 'y', nbins = 4)
	ax4.set_ylabel("Least Square error", {'rotation':-90})
	ax3.set_ylabel("Ranking value")
	ax3.set_xlabel("Fit to Choice <-> Fit to RT")
	# ax3.set_xticks([])		
	ax3.get_xaxis().set_tick_params(direction='out')
	ax3.set_xticks([np.argmin(value[s]['tche']), np.argmin(value[s]['test'])])
	ax3.set_xticklabels(['R', 'L'])




# line2 = tuple([Line2D(range(1),range(1),alpha=1.0,color=colors_m[m], linewidth = 2) for m in colors_m.keys()])
# figlegend(line2,tuple(legend_m.values()), loc = 'lower right', bbox_to_anchor = (0.95, 0.05))
# line3 = tuple([Line2D(range(1),range(1), linestyle = '', marker = markers[i], alpha=1.0, markerfacecolor = 'white', color='black') for i in xrange(len(markers))])
# figlegend(line3,tuple(["Variation "+str(i+1) for i in xrange(5)]), loc = 'lower right', bbox_to_anchor = (0.78, 0.15))



savefig('fig6_monkeys_pareto.pdf', dpi = 900, facecolor= 'white', bbox_inches = 'tight')
os.system("evince fig6_monkeys_pareto.pdf &")


