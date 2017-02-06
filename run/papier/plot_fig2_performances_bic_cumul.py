
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
                'random':'white'})

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
# for s in ['p']:
	# MONKEYS PERFORMANCE	
	data[s] = np.genfromtxt("../../data/data_txt_3_repeat/"+s+".txt", dtype = 'int')
	performance_monkeys[s] = []	
	problems_sar = []
	tmp = [[1, data[s][0,4]-1,data[s][0,5]-1,data[s][0,3]]]	
	count = 0
	tmp2 = []
	tmp3 = []
	count2 = 0
	for i in xrange(1,len(data[s])):				
		if data[s][i-1,1] != data[s][i,1] or int(data[s][i,2]) == 0 and int(data[s][i-1,2]) == 1: # new problem 		
			# Check if the last problem have repeat phase
			tmp = np.array(tmp)
			if tmp[:,3].sum()>=1 and i - count > 1:
				problems_sar += list(tmp)				
				performance_monkeys[s].append(tmp3)
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
	model.test_call(problems_sar, p_test_v1[s]['best_choice'][m+"_"+str(np.max(bic.keys()))])
	performance_models[s] = np.array(model.performance)



figure()
for s,i in zip(monkeys.keys(),range(1,6)):
	subplot(2,3,i)
	x = np.vstack(np.arange(1,len(performance_monkeys[s])+1))
	monkey = np.cumsum(performance_monkeys[s],0).astype('float')/x
	modell = np.cumsum(performance_models[s],0).astype('float')/x
	colors = {1:'blue', 2:'red', 3:'green'}
	for j in xrange(3):
		plot(monkey[:,j], '--', color = colors[j+1], label = str(j))
		plot(modell[:,j], '-', color = colors[j+1])	
	
	# ylim(0.7,1)
	legend()
	title(s+" "+best_model[s])


show()



