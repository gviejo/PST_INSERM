
import numpy as np 
from pylab import *
import cPickle as pickle
import sys, os
sys.path.append("../sferes/last_set_models/")
from fusion_1 import fusion_1
from mixture_1 import mixture_1
from bayesian_1 import bayesian_1
from qlearning_1 import qlearning_1

def center(array):
	array = array - np.median(array)
	array = array / (np.percentile(array, 75) - np.percentile(array, 25))
	return array


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
time_monkeys = {}
time_models = {}
best_model = {}


for s in monkeys.keys():
	# MONKEYS PERFORMANCE AND REACTION TIME
	data[s] = np.genfromtxt("../../data/data_txt_3_repeat/"+s+".txt", dtype = 'float')
	data[s][:,6] = center(data[s][:,6])
	performance_monkeys[s] = []
	time_monkeys[s] = {i:[] for i in xrange(1,6)}
	time_models[s] = {i:[] for i in xrange(1,6)}
	problems_sar = []
	tmp = [[1, data[s][0,4]-1,data[s][0,5]-1,data[s][0,3]]]	
	count = 0	
	tmp3 = [] # for choices
	tmp4 = [data[s][0,6]] # for rt
	count2 = 0
	length_problems_count = []
	for i in xrange(1,len(data[s])):				
		if data[s][i-1,1] != data[s][i,1] or int(data[s][i,2]) == 0 and int(data[s][i-1,2]) == 1: # new problem 		
			# Check if the last problem have repeat phase
			tmp = np.array(tmp)
			if tmp[:,3].sum()>=1 and i - count > 1 and len(tmp4) > len(tmp):
				problems_sar += list(tmp)				
				performance_monkeys[s].append(tmp3)
				time_monkeys[s][len(tmp4[0:-3])].append(tmp4)
				length_problems_count.append(len(tmp))
				count2+=1
			# new problem 
			tmp = [[1, data[s][i,4]-1,data[s][i,5]-1,data[s][i,3]]]						
			tmp4 = [data[s][i,6]]
			count = i 
			# print i, data[s][i,0], "problem"
		else :
			if int(data[s][i,2]) == 0 and int(data[s][i-1,2]) == 0: # search phase				
				tmp.append([0, data[s][i,4]-1,data[s][i,5]-1,data[s][i,3]])
				tmp4.append(data[s][i,6])
				# print i, data[s][i,0], "seach"
			elif int(data[s][i,2]) == 1 and int(data[s][i-1,2]) == 0:# repeat phase					
				tmp3 = data[s][i:i+3,3]
				tmp4+=list(data[s][i:i+3,6])				
				# print i, data[s][i,0], "repeat"
		
		if len(performance_monkeys[s]):
			if len(performance_monkeys[s][-1]) != 3:
				performance_monkeys[s].pop(-1)


	performance_monkeys[s] = np.array(performance_monkeys[s])	
	problems_sar = np.array(problems_sar)
	for k in time_monkeys[s].keys():
		time_monkeys[s][k] = np.array(time_monkeys[s][k])
	
	# MODELS PERFORMANCE AND REACTION TIMES
	
	m = p_test_v1[s]['best_tche'].keys()[0]
	model = models[m][1]
	best_model[s] = m
	model.test_call(2, problems_sar, p_test_v1[s]['best_tche'][m])
	performance_models[s] = np.array(model.performance)
	
	
	########################################
	# separating performances according to length of the search phase
	# mean performances according to trials
	# mean rt according to to trials

	########################################	
	length_problems_count = np.array(length_problems_count)
	# centering rt from models
	for k in xrange(len(model.timing)): # repeat
		tmp = []
		for i in xrange(len(length_problems_count)):
			tmp.append(model.timing[k][i,0:length_problems_count[i]+3]) # blocs
		tmp = np.hstack(np.array(tmp))
		model.timing[k] = model.timing[k] - np.median(tmp)
		model.timing[k] = model.timing[k] / (np.percentile(tmp, 75) - np.percentile(tmp, 25))
	tmp = {}
	tmp2 = {}
	for i in np.unique(length_problems_count):
		# monkey
		index = length_problems_count == i
		tmp[i] = np.array([np.mean(performance_monkeys[s][index], 0),
							np.var(performance_monkeys[s][index], 0)])
		tmp2[i] = np.array([performance_models[s][:,index,:].reshape(len(performance_models[s])*index.sum(),3).mean(0),
							performance_models[s][:,index,:].reshape(len(performance_models[s])*index.sum(),3).var(0)])

		time_monkeys[s][i] = np.array([np.mean(time_monkeys[s][i], 0),
										np.var(time_monkeys[s][i], 0)])
		time_models[s][i] = np.array([np.mean(model.timing[:,index,0:i+3].reshape(len(model.timing)*index.sum(),i+3), 0),
										np.var(model.timing[:,index,0:i+3].reshape(len(model.timing)*index.sum(),i+3), 0)])

	performance_models[s] = tmp2
	performance_monkeys[s] = tmp


	





figure(figsize = (15,10))
count = 1
for s in monkeys.keys():
	subplot(5,2,count)
	xpos = []
	for t in range(1,6):		
		x = np.arange(3) + (t-1)*2.4
		plot(x, performance_monkeys[s][t][0], 'o--', color = 'black', linewidth = 2)
		fill_between(x, performance_monkeys[s][t][0]-performance_monkeys[s][t][1],
						performance_monkeys[s][t][0]+performance_monkeys[s][t][1],
						 linewidth = 0, 
						 edgecolor = None,
						 facecolor = 'black',
						 alpha = 0.5,
						 visible = True)						 
		plot(x, performance_models[s][t][0], 'o-', color = colors_m[best_model[s]])
		fill_between(x, performance_models[s][t][0]-performance_models[s][t][1],
						performance_models[s][t][0]+performance_models[s][t][1],
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

	subplot(5,2,count)
	xpos = [-1]
	for t in xrange(1,6):		
		x = np.arange(xpos[-1]+1, xpos[-1]+1+t+3)
		plot(x, time_monkeys[s][t][0], 'o--', color = 'black', linewidth =2)
		fill_between(x, time_monkeys[s][t][0]-time_monkeys[s][t][1],
						time_monkeys[s][t][0]+time_monkeys[s][t][1],						
						linewidth = 0, 
						edgecolor = None,
						facecolor = 'black',
						alpha = 0.5)						 
		plot(x, time_models[s][t][0], 'o-', color = colors_m[best_model[s]], linewidth = 2)
		fill_between(x, time_models[s][t][0]-time_models[s][t][1],
						time_models[s][t][0]+time_models[s][t][1],						
						linewidth = 0, 
						edgecolor = None,
						facecolor = colors_m[best_model[s]], 
						alpha = 0.5)
		xpos.append(x[-1])
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
savefig("fig2_choice_rt_v1_tche.pdf", dpi = 900, facecolor = 'white', bbox_inches = 'tight')

os.system("evince fig2_choice_rt_v1_tche.pdf")





