
import numpy as np 
from pylab import *
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


for s in monkeys.keys():
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
	problems_sar = np.array(problems_sar)

	# MODELS PERFORMANCE
	# best model for s with bic choice only
	bic = {float(i.split("_")[1]):i.split("_")[0] for i in p_test_v1[s]['best_choice'].keys()}
	m = bic[np.max(bic.keys())]		
	model = models[m][1]
	best_model[s] = m
	model.test_call(5, problems_sar, p_test_v1[s]['best_choice'][m+"_"+str(np.max(bic.keys()))])
	performance_models[s] = np.array(model.performance)
	
	
	########################################
	# separating performances according to length of the search phase
	# mean performances according to trials
	########################################	
	length_problems_count = np.array(length_problems_count)
	tmp = {}
	tmp2 = {}
	for i in np.unique(length_problems_count):
		index = length_problems_count == i
		tmp[i] = np.array([np.mean(performance_monkeys[s][index], 0),
							np.var(performance_monkeys[s][index], 0)])
		tmp2[i] = np.array([performance_models[s][:,index,:].reshape(len(performance_models[s])*index.sum(),3).mean(0),
							performance_models[s][:,index,:].reshape(len(performance_models[s])*index.sum(),3).var(0)])


	performance_models[s] = tmp2
	performance_monkeys[s] = tmp


figure(figsize = (15,10))
count = 1
for s in monkeys.keys():
	subplot(5,1,count)
	xpos = []
	for t in range(1,6):		
		x = np.arange(3) + (t-1)*2.4
		fill_between(x, performance_monkeys[s][t][0],
						performance_monkeys[s][t][0]-performance_monkeys[s][t][1],
						performance_monkeys[s][t][0]+performance_monkeys[s][t][1],
						 'o--',
						 linewidth = 0, 
						 edgecolor = None,
						 facecolor = 'black',
						 alpha = 0.5,
						 visible = True)						 
		fill_between(x, performance_models[s][t][0],
						performance_models[s][t][0]-performance_models[s][t][1],
						performance_models[s][t][0]+performance_models[s][t][1],
						'o-', 
						linewidth = 0, 
						edgecolor = None,
						facecolor = colors_m[best_model[s]], 
						alpha = 0.5)
		xpos.append(x[1])
	xticks(xpos, tuple([str(i)+" errors" for i in xrange(5)]))	
	locator_params(axis='y',nbins=3)
	title("Monkey "+s)	
	if count == 3:
		ylabel("Performances in repetition (%)", size = 20, labelpad = 30)
	count += 1	
	grid()
	legend()

line2 = tuple([Line2D(range(1),range(1), linestyle = style[m], alpha=1.0, color=colors_m[m], linewidth = 4) for m in ['fusion', 'mixture', 'monkeys']])
legend(line2,tuple(['Entropy-based coordination', 'Weight-based mixture', 'Monkey']), loc = 'upper center', bbox_to_anchor = (0.5, 6.25), fancybox = True, shadow = True, ncol = 3)
savefig("fig2_bic_choice.pdf", dpi = 900, facecolor = 'white', bbox_inches = 'tight')




