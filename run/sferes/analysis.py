#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

load and and plot multi objective results from Sferes 2 optimisation 


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import sys
import os
from optparse import OptionParser
import numpy as np

sys.path.append("../../src")

from Models import FSelection

from matplotlib import *
from pylab import *

from Sferes import pareto
from itertools import *
from time import sleep
import cPickle as pickle

# -----------------------------------
# ARGUMENT MANAGER
# -----------------------------------
if not sys.argv[1:]:
    sys.stdout.write("Sorry: you must specify at least 1 argument")
    sys.stdout.write("More help avalaible with -h or --help option")
    sys.exit(0)
parser = OptionParser()
parser.add_option("-i", "--input", action="store", help="The name of the directory to load", default=False)
parser.add_option("-m", "--model", action="store", help="The name of the model to test \n If none is provided, all files are loaded", default=False)
parser.add_option("-o", "--output", action="store", help="The output file of best parameters to test", default=False)
(options, args) = parser.parse_args()
# -----------------------------------

# -----------------------------------
# LOADING DATA
# -----------------------------------
front = pareto(options.input)
front.constructParetoFrontier('bic')
front.constructMixedParetoFrontier()
front.rankTchebytchev()
front.rankOWA()
front.rankDistance()
front.retrieveRanking()
front.evaluate()
front.writePlot(options.input)
front.writeParameters(options.input)

os.system("evince "+options.input+"_*.pdf")
os.system("pdftk "+options.input+"_pareto_front.pdf "+options.input+"_mixed_pareto_front.pdf "+options.input+"_evaluation_sferes_call.pdf cat output "+options.input+"_all_plot.pdf")

with open(options.input+"_best_parameters.pickle", 'wb') as f:
	pickle.dump(front.p_test, f)

