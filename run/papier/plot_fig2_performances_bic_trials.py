
import numpy as np 
from pylab import *
import matplotlib.gridspec as gridspec
import cPickle as pickle
import sys, os
sys.path.append("../sferes/last_set_models/")
from fusion_1 import fusion_1
from mixture_1 import mixture_1
from bayesian_1 import bayesian_1
from qlearning_1 import qlearning_1




monkeys =   { 'g':12701,
              'm':34752,
              'p':27692,
              'r':11634,
              's':13348 }

models_to_id = dict({	'fusion':1,
						'mixture':2,
						'bayesian':3,
						'qlearning':4})
models = dict({'fusion':{1:fusion_1()},
				'mixture':{1:mixture_1()},
				'bayesian':{1:bayesian_1()},
				'qlearning':{1:qlearning_1()}
			})

colors_m = dict({'fusion':'#F1433F',
                'bayesian':'#D5A253',
                'qlearning': '#6E8243',
                # 'selection':'#70B7BA',
                'mixture':'#3D4C53',
                'random':'white',
                'monkeys':'black'})
style = dict({'fusion':'-',
				'mixture':'-',
				'monkeys':'--'})

legend_m = dict({'fusion':'Entropy-based coordination',
                'bayesian':'Bayesian Working Memory',
                'qlearning':'Q-Learning',
                # 'selection': 'VPI-based selection',
                'mixture': 'Weight-based mixture',
                'random': 'Random choice'})


with open("p_test_v1.pickle", 'rb') as f:
	p_test_v1 = pickle.load(f)

data = {}
performance_monkeys = {}
performance_models = {}
best_model = {}
length_monkeys = {}
length_models = {}



for s in monkeys.keys():
	###########################################################################################
	# MONKEYS PERFORMANCE	
	data[s] = np.genfromtxt("../../data/data_txt_3_repeat/"+s+".txt", dtype = 'int')
	performance_monkeys[s] = []	
	problems_sar = []
	tmp = [[1, data[s][0,4]-1,data[s][0,5]-1,data[s][0,3]]]	
	count = 0	
	tmp3 = []
	count2 = 0
	length_problems_count = []
	for i in xrange(1,len(data[s])):				
		if data[s][i-1,1] != data[s][i,1] or int(data[s][i,2]) == 0 and int(data[s][i-1,2]) == 1: # new problem 		
			# Check if the last problem have repeat phase
			tmp = np.array(tmp)
			if tmp[:,3].sum()>=1 and i - count > 1:
				problems_sar += list(tmp)				
				performance_monkeys[s].append(tmp3)
				length_problems_count.append(len(tmp))
				count2+=1
			# new problem 
			tmp = [[1, data[s][i,4]-1,data[s][i,5]-1,data[s][i,3]]]						
			count = i 
			# print i, "problem"
		else :
			if int(data[s][i,2]) == 0 and int(data[s][i-1,2]) == 0: # search phase				
				tmp.append([0, data[s][i,4]-1,data[s][i,5]-1,data[s][i,3]])
				# print i, "seach"
			elif int(data[s][i,2]) == 1 and int(data[s][i-1,2]) == 0:# repeat phase					
				tmp3 = data[s][i:i+3,3]
				# performance_monkeys[s].append(data[s][i:i+3,3])
				# print i, "repeat"
		
		if len(performance_monkeys[s]):
			if len(performance_monkeys[s][-1]) != 3:
				performance_monkeys[s].pop(-1)
	performance_monkeys[s] = np.array(performance_monkeys[s])
	length_problems_count = np.array(length_problems_count)
	tmp = {}
	for i in np.unique(length_problems_count):
		index = length_problems_count == i
		tmp[i] = np.array([np.mean(performance_monkeys[s][index], 0),
							np.var(performance_monkeys[s][index], 0)])
	performance_monkeys[s] = tmp
	length_monkeys[s] = np.array([np.sum(length_problems_count==i) for i in xrange(1, 20)]).astype('float')
	length_monkeys[s] = length_monkeys[s]/np.sum(length_monkeys[s])

	##########################################################################################
	# MODEL PERFORMANCE
	problems_sar = np.array(problems_sar)
	# best model for s with bic choice only
	bic = {float(i.split("_")[1]):i.split("_")[0] for i in p_test_v1[s]['best_choice'].keys()}
	m = bic[np.max(bic.keys())]		
	model = models[m][1]
	best_model[s] = m
	model.test_call(1000, problems_sar, p_test_v1[s]['best_choice'][m+"_"+str(np.max(bic.keys()))])
	performance_models[s] = np.array(model.performance)
	tmp2 = {int(i):[] for i in np.unique(model.length)}
	for i in xrange(performance_models[s].shape[0]):		
		for j in np.unique(model.length[i]):
			index = model.length[i] == int(j)
			tmp2[int(j)].append(model.performance[i,index])
			# tmp2[i].append() = np.array([performance_models[s][:,index,:].reshape(len(performance_models[s])*index.sum(),3).mean(0),
			# 					performance_models[s][:,index,:].reshape(len(performance_models[s])*index.sum(),3).var(0)])
	for i in tmp2.iterkeys():
		tmp2[i] = np.vstack(tmp2[i])
		tmp2[i] = np.array([np.mean(tmp2[i], 0),
							np.var(tmp2[i], 0)])
	performance_models[s] = tmp2
	length_models[s] = np.array([np.sum(model.length==i) for i in xrange(1, 20)]).astype('float')
	length_models[s] = length_models[s]/np.sum(length_models[s])

















figure(figsize = (13,5))
subplots_adjust(hspace = 0.7, wspace = 0.3)
outer = gridspec.GridSpec(3, 2) 
count = 1
alpha = 0.2
for s in monkeys.keys():
	gs = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec = outer[monkeys.keys().index(s)])
	ax = subplot(gs[0])		
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax2 = ax.twinx()
	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_bounds(0.0, 0.3)
	
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	xpos = [-1]
	xtick_pos = []	
	for t in range(1,6):		
		# x = np.arange(xpos[-1]+1, xpos[-1]+1+t+3)
		x = np.arange(3) + (t-1)*2.9		
		ax.plot(x[-3:], performance_monkeys[s][t][0], 'o--', linewidth = 2, color = 'black')		
		ax.fill_between(x[-3:], performance_monkeys[s][t][0]-performance_monkeys[s][t][1],
						performance_monkeys[s][t][0]+performance_monkeys[s][t][1],
						linewidth = 0.5, 
						edgecolor = 'black',
						facecolor = 'black',
						alpha = alpha)
		ax.plot(x[-3:], performance_models[s][t][0], 'o-', linewidth = 2, color = colors_m[best_model[s]])
		ax.fill_between(x[-3:], performance_models[s][t][0]-performance_models[s][t][1],
						performance_models[s][t][0]+performance_models[s][t][1],
						linewidth = 0.5, 
						edgecolor = colors_m[best_model[s]],
						facecolor = colors_m[best_model[s]], 
						alpha = alpha)		
		xpos.append(x[1])
		xtick_pos.append(x[-2])
		ax2.bar(x[-2], length_monkeys[s][t-1], 0.5, color = 'white', edgecolor = 'black', linewidth = 2.0, linestyle = '-', alpha = 0.8, hatch = '///')
		ax2.bar(x[-1]-0.5, length_models[s][t-1], 0.5, color = colors_m[best_model[s]], edgecolor = 'black', alpha = 0.8)


	# ax2.set_ylim(0,)
	xticks(xtick_pos, tuple([str(i)+" Err" for i in xrange(5)]))	
	ax2.set_ylim(0,0.8)
	ax.locator_params(axis='y',nbins=3)	
	# ax2.locator_params(axis='y',nbins=2)	
	ax2.set_yticks([0, 0.3])
	ax2.set_yticklabels(("0", ".3"), fontsize = 8)
	ax.set_ylabel("Accuracy")
	ax2.set_ylabel("Density", {'fontsize':9,'rotation':-90})
	ax2.yaxis.set_label_coords(1.08, 0.2)
	# ax.annotate('Monkey '+s, (0.0,0.0), textcoords= 'axes fraction', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
	title('Monkey '+s)

		
	count = count + 1 if count%2 else count + 3


line2 = tuple([Line2D(range(1),range(1), linestyle = style[m], alpha=1.0, color=colors_m[m], linewidth = 4) for m in ['fusion', 'mixture', 'monkeys']])
legend(line2,tuple(['Entropy-based coordination', 'Weight-based mixture', 'Monkey']), loc = 'upper center', bbox_to_anchor = (1.7, 1.1), fancybox = False)

savefig("fig2_bic_choice.pdf", dpi = 900, facecolor = 'white', bbox_inches = 'tight')
os.system("evince fig2_bic_choice.pdf &")

