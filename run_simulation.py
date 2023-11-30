import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import time
from numba import jit
import csv
from function_collection import *

'''
INITIALIZATION

Run to initialize jit for the translation computationally-intensive Python function to machine code using Numba.
'''

print("Initializing jit")
random.seed(0)
weights = create_rand_weights(neurons)
print(updating_steps_sparse(weights, A_pattern, 1000))
print("Done.")

np.set_printoptions(threshold=sys.maxsize)


'''
SIMULATION

Run to reproduce the results illustrated in the paper.

Letter patterns and required functions are imported from function_collection.py.

You can specify different parameters in the inputs to present_pattern_cum or, 
to change properties of the repertoires, at the end of function_collection.py.

Two files are saved. For every 100/10 repetitions, are line is added.
- abs_conv.csv tracks the number of abstraction groups which recognize a given letter (hence 26 columns).
- groups_conv.csv tracks the number of recognition groups which recognize a given letter (again, 26 columns).
'''

reps = 10001

stepnos = []

for i in range(reps):
    print("Rep: " + str(i))
    rand = random.sample(range(26), 1)[0]
    print("Selected pattern: " + str(rand))
    for lettrep in range(10):
        print("Presented letter " + str(lettrep+1) + " time(s).")
        groups, cutoff, states, max_groups, stepno = present_pattern_cum(groups, con_groups, letter_patterns[rand], refrac = 1.2, att_factor=0.97, theta=0.95,
                                                                     learning_rate=0.05, cutoff_only=0)
        stepnos.append(stepno)
        abs_overlap = []
        abs_updated = 0
        for j in range(len(abs_groups)):
            abs_overlap.append(len(intersection(max_groups, abs_con_groups[j])))
            abs_updated = abs_updated + abs_weight_updating_cum(j, cutoff, states, abs_groups, abs_con_groups, beta=0.3)
        print(abs_overlap)
        print("Abs groups updated: " + str(abs_updated))
    if i%10 == 0:
        conv_no = []
        for k in range(len(letter_patterns)):
            res = conv_achieved(abs_groups, letter_patterns[k])
            conv_no.append(res)
        if i%100 == 0:
            lett_conv_no = []
            for j in range(len(letter_patterns)):
                print(j)
                lett_res = conv_achieved(groups, letter_patterns[j])
                lett_conv_no.append(lett_res)
        with open('abs_conv.csv', 'a+') as file:
            write = csv.writer(file)
            write.writerow(conv_no)
        with open('groups_conv.csv', 'a+') as file2:
            write2 = csv.writer(file2)
            write2.writerow(lett_conv_no)
