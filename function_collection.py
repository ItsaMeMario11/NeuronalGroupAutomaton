import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import time
from numba import jit

# SETUP

dimx = 4
dimy = 4

neurons = dimx * dimy

min_steps = neurons
max_steps = 2 * neurons




# LETTER PATTERNS

A_pattern = np.array([[0, 1, 1, 0],
                      [1, 0, 0, 1],
                      [1, 1, 1, 1],
                      [1, 0, 0, 1]])
B_pattern = np.array([[1, 1, 1, 0],
                      [1, 1, 1, 1],
                      [1, 0, 0, 1],
                      [1, 1, 1, 0]])
C_pattern = np.array([[1, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]])
D_pattern = np.array([[1, 1, 1, 0],
                      [1, 0, 0, 1],
                      [1, 0, 0, 1],
                      [1, 1, 1, 0]])
E_pattern = np.array([[1, 1, 1, 1],
                      [1, 1, 1, 0],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]])
F_pattern = np.array([[1, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 1, 0],
                      [1, 0, 0, 0]])
G_pattern = np.array([[1, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 0, 0, 1],
                      [1, 1, 1, 1]])
H_pattern = np.array([[1, 0, 0, 1],
                      [1, 0, 0, 1],
                      [1, 1, 1, 1],
                      [1, 0, 0, 1]])
I_pattern = np.array([[0, 1, 1, 1],
                      [0, 0, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 1, 1]])
J_pattern = np.array([[0, 0, 1, 1],
                      [0, 0, 0, 1],
                      [1, 0, 0, 1],
                      [0, 1, 1, 0]])
K_pattern = np.array([[1, 0, 0, 1],
                      [1, 0, 1, 0],
                      [1, 1, 1, 0],
                      [1, 0, 0, 1]])
L_pattern = np.array([[1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1]])
M_pattern = np.array([[1, 0, 1, 1],
                      [1, 1, 0, 1],
                      [1, 0, 0, 1],
                      [1, 0, 0, 1]])
N_pattern = np.array([[1, 0, 0, 1],
                      [1, 1, 0, 1],
                      [1, 0, 1, 1],
                      [1, 0, 0, 1]])
O_pattern = np.array([[1, 1, 1, 1],
                      [1, 0, 0, 1],
                      [1, 0, 0, 1],
                      [1, 1, 1, 1]])
P_pattern = np.array([[1, 1, 1, 0],
                      [1, 0, 0, 1],
                      [1, 1, 1, 0],
                      [1, 0, 0, 0]])
Q_pattern = np.array([[1, 1, 1, 1],
                      [1, 0, 0, 1],
                      [1, 0, 1, 1],
                      [1, 1, 1, 0]])
R_pattern = np.array([[1, 1, 1, 0],
                      [1, 0, 0, 1],
                      [1, 1, 1, 0],
                      [1, 0, 1, 1]])
S_pattern = np.array([[1, 1, 1, 1],
                      [1, 0, 0, 0],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]])
T_pattern = np.array([[1, 1, 1, 1],
                      [0, 0, 1, 0],
                      [0, 0, 1, 0],
                      [0, 0, 1, 0]])
U_pattern = np.array([[1, 0, 0, 1],
                      [1, 0, 0, 1],
                      [1, 0, 0, 1],
                      [1, 1, 1, 1]])
V_pattern = np.array([[1, 0, 0, 1],
                      [1, 0, 0, 1],
                      [1, 0, 0, 1],
                      [0, 1, 1, 0]])
W_pattern = np.array([[1, 0, 0, 1],
                      [1, 0, 0, 1],
                      [1, 0, 1, 1],
                      [1, 1, 0, 1]])
X_pattern = np.array([[1, 0, 1, 1],
                      [1, 1, 1, 0],
                      [0, 1, 1, 1],
                      [1, 1, 0, 1]])
Y_pattern = np.array([[1, 0, 0, 1],
                      [0, 1, 1, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0]])
Z_pattern = np.array([[1, 1, 1, 1],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [1, 1, 1, 1]])

letter_patterns = [A_pattern, B_pattern, C_pattern, D_pattern, E_pattern, F_pattern, G_pattern, H_pattern, I_pattern,
                   J_pattern, K_pattern, L_pattern, M_pattern, N_pattern, O_pattern, P_pattern, Q_pattern, R_pattern,
                   S_pattern, T_pattern, U_pattern, V_pattern, W_pattern, X_pattern, Y_pattern, Z_pattern]



# GENERAL

# These functions code the convergence of a Hopfield network
@jit(nopython=True)
def updating_steps_sparse(weights, input_pattern, prev_max, print_pat = 0):
  count = 0
  steps = 0

  input = input_pattern.reshape(-1).copy().astype(np.float64)

  while count < neurons and steps <= prev_max:
    for i in range(neurons):
      if np.dot(weights[i], input) < 0:
        control = input[i]
        input[i] = 0
        steps = steps + 1
        if control == input[i]:
          count = count + 1
        else:
          count = 0
      else:
        control = input[i]
        input[i] = 1
        steps = steps + 1
        if control == input[i]:
          count = count + 1
        else:
          count = 0
  if steps < prev_max:
    new_max = steps
  else:
    new_max = prev_max
  if print_pat == 0:
    return np.float32(steps), np.float32(new_max)


def updating(input_pattern, weights):
  count = 0
  steps = 0

  input = input_pattern.reshape(-1).copy()

  while count < neurons:
    for i in range(neurons):
      if np.dot(weights[i], input) < 0:
        control = input[i]
        input[i] = 0
        steps = steps + 1
        if control == input[i]:
          count = count + 1
        else:
          count = 0
      else:
        control = input[i]
        input[i] = 1
        steps = steps + 1
        if control == input[i]:
          count = count + 1
        else:
          count = 0

  output = input.reshape(input_pattern.shape[0], input_pattern.shape[1])
  return output, steps

# These functions create a random weight matrix of a given size
def create_upper_matrix(values, size):
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 0)] = values
    return upper


def create_rand_weights(size):
    triang = int((size * size - size) / 2 + size)  # Calculates triangular number needed to fill upper triangular matrix

    upper = create_upper_matrix(random.choices([-1, 1], k=triang), size)
    full = upper + np.transpose(upper)
    np.fill_diagonal(full, 0)
    return full

# This function find the inversion of a pattern (which has identical energy to the positive and is indistinguishable from it). Used for testing.
def complement_pattern(pat):
    rows = pat.shape[0]
    cols = pat.shape[1]
    raw = pat.reshape(-1)
    new_raw = np.zeros(len(raw))
    for i in range(len(raw)):
        if raw[i] == 0:
            new_raw[i] = 1
    new = new_raw.reshape(rows, cols)
    return new

# Creates a neuronal group with similar weights. Used to make topographical organization computationally efficient.
def create_sim_weights(ref_weights, perc_resamp):
    size = ref_weights.shape[0]
    raw_weights = ref_weights[np.triu_indices(size, k=1)]  # Extract upper triangular numbers
    n_resamp = int(np.ceil(perc_resamp * len(
        raw_weights)))  # Determine number of resampled weights (ceiling of perc_resamp*no of upper triangular numbers)
    ind_resamp = random.sample(range(len(raw_weights)), n_resamp)  # Sample indices to be resampled
    for i in range(n_resamp):
        raw_weights[ind_resamp[i]] = random.sample([-1, 1], 1)[0]  # Resample (between 0 and n_resamp)
    new_weights = np.zeros((size, size))
    new_weights[np.triu_indices(size, 1)] = raw_weights
    new_weights = new_weights + np.transpose(new_weights)  # Turned back into weight matrix
    return new_weights

# Finds overlap between two lists (auxiliary function)
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3



# REPERTOIRES

# Creates the recognition repertoire
def create_rec_rep(n_expansions, n_startgroups_per_iter, n_groups_per_iter, percentage_cross_connections,
                   percentage_resamp_per_group):
    print("Creating recognition repertoire.")

    last = 0

    # Calculate total number of groups
    group_no = 1 + np.sign(n_expansions) * n_groups_per_iter + max(n_expansions - 1,
                                                                   0) * n_groups_per_iter * n_startgroups_per_iter
    print(group_no)
    groups = np.zeros((group_no, neurons, neurons))
    con_groups = [[] for i in range(group_no)]
    # print(con_groups[0])

    # Create first neuronal group
    groups[0] = create_rand_weights(neurons)

    # Create first expansion
    if n_expansions > 0:
        last_bef_exp = last
        for i in range(n_groups_per_iter):
            con_groups[0].append(i + 1)
            groups[i + 1] = create_sim_weights(groups[0], percentage_resamp_per_group)
            last = last + 1
            con_groups[i + 1].append(0)

    # print(last_bef_exp)
    # print(last)

    # Create remaining expansions
    if n_expansions > 1:
        for exp in range(2, n_expansions + 1):
            print("Creating expansion " + str(exp))
            # print(range(last_bef_exp+1, last+1))
            expand_ind = random.sample(range(last_bef_exp + 1, last + 1), n_startgroups_per_iter)
            # print(expand_ind)
            last_bef_exp = last
            for j in range(len(expand_ind)):
                for k in range(n_groups_per_iter):
                    # print("expand_ind[j]: " + str(expand_ind[j]))
                    con_groups[expand_ind[j]].append(last + 1)
                    groups[last + 1] = create_sim_weights(groups[expand_ind[j]], percentage_resamp_per_group)
                    last = last + 1
                    con_groups[last].append(expand_ind[j])

            # print(last)
            # print(last_bef_exp)

            # Create cross connections
            if exp > 2:
                n_cross = int(np.round(percentage_cross_connections * (last - last_bef_exp)))
                rand1 = random.sample(range(last_bef_exp + 1, last + 1), n_cross)
                # print(rand1)
                rand2 = random.sample(
                    range(last_bef_exp - n_groups_per_iter * n_startgroups_per_iter + 1, last_bef_exp + 1), n_cross)
                # print(rand2)
                for m in range(len(rand1)):
                    con_groups[rand1[m]].append(rand2[m])
                    con_groups[rand2[m]].append(rand1[m])

    print("Done.")

    return groups, con_groups

# Creates the abstraction repertoire
def create_abs_rep(groups, con_groups, nabs, extend=1):

    print("Creating abstraction repertoire.")

    n = groups.shape[0]
    abs_centers = random.sample(range(n), k=nabs)
    abs_con_groups = []
    for i in range(nabs):
        abs_con_groups.append(con_groups[abs_centers[i]])

    if extend >= 1:
        for j in range(nabs):
            for k in range(len(abs_con_groups[j])):
                abs_con_groups[j].extend(con_groups[abs_con_groups[j][k]])

    if extend >= 2:
        for j in range(nabs):
            for k in range(len(abs_con_groups[j])):
                abs_con_groups[j].extend(con_groups[abs_con_groups[j][k]])

    for l in range(nabs):
        abs_con_groups[l] = list(dict.fromkeys(abs_con_groups[l]))

    abs_groups = np.zeros((nabs, neurons, neurons))

    for o in range(nabs):
        abs_groups[o] = create_rand_weights(neurons)

    print("Done.")

    return abs_groups, abs_con_groups



# PATTERN PRESENTING

# The functions `create_rec_rep` and `create_abs_rep` create the four tensors `groups`, `con_groups`,
# `abs_groups`, and `abs_con_groups`. These are the input for the pattern presenting function.

# The following functions are auxiliary functions to determine excitation thresholds in a sensible manner.
def phi(steps):
    return 1 / steps


def congroup_excitation(absno, states, abs_con_groups):
    congroup_states = []
    for i in range(len(abs_con_groups[absno])):
        congroup_states.append(states[abs_con_groups[absno][i]])
    return sum(congroup_states)


def max_excitation_cum(absno, abs_con_groups, att_factor=0.95, thetaA=0.2):
    geom_conv = phi(max_steps) * (1 / (1 - att_factor))
    ncon = len(abs_con_groups[absno])
    max = (phi(min_steps) + geom_conv) * ncon
    #print("max: " + str(max))
    threshold = (thetaA * phi(min_steps) + geom_conv) * ncon
    return threshold


def abs_weight_updating_cum(absno, cutoff, states, abs_groups, abs_con_groups, beta=0.3):
    threshold = max_excitation_cum(absno, abs_con_groups)
    #print("thr: " + str(threshold))
    con = congroup_excitation(absno, states, abs_con_groups)
    #print("con: " + str(con))
    if con > threshold:
        for i in range(len(abs_con_groups[absno])):
            if states[abs_con_groups[absno][i]] > cutoff:
                sum = abs_groups[absno] + beta * (states[abs_con_groups[absno][i]] - cutoff) * groups[
                    abs_con_groups[absno][i]]
                abs_groups[absno] = sum / np.max(np.absolute(sum))
        # print("Updated abs group " + str(absno))
        return 1
    else:
        return 0
    #    print("No abs group updated.")

# This function presents a pattern to all groups in the recognition repertoire.
def present_pattern_cum(groups, con_groups, in_pattern, refrac = 1.5, theta=0.25, learning_rate=0.1, att_factor=0.95, cutoff_only=0):
    n = len(groups)
    stepno = []
    max_steps = neurons * 10
    for i in range(n):
        steps, max_steps = updating_steps_sparse(groups[i], in_pattern, max_steps)
        stepno.append(steps)
        states[i] = phi(steps) + att_factor * states[i]

    max_state = states[np.argmax(states)]
    print(max_state)
    geom_conv = phi(max_steps) * (1 / (1 - att_factor)) # convergence of excitation state of group that never recognizes pattern (geometric series)
    cutoff = theta * max_state
    if cutoff_only == 0:
        max_groups = [j for j in range(len(states)) if states[j] > cutoff]
        # print(max_groups)
        print(len(max_groups))
        nmax = len(max_groups)

        act_upd_tot = 0
        for k in range(nmax):
            act_upd = []
            for l in range(len(con_groups[max_groups[k]])):
                if states[con_groups[max_groups[k]][l]] <= refrac * geom_conv: # no updating if group more excited (by factor refrac) than long-term value of group that never recognizes pattern; indicates useful activation
                    sum = groups[con_groups[max_groups[k]][l]] + learning_rate * (states[max_groups[k]] - cutoff) * \
                          groups[max_groups[k]]
                    groups[con_groups[max_groups[k]][l]] = sum / np.max(np.absolute(sum))
                    act_upd.append(1)
            #print(str(len(con_groups[max_groups[k]])) + " con groups but only updated " + str(np.sum(act_upd)) + " of them.")
            # print("Actually updated: " + str(np.sum(act_upd)))
            act_upd_tot = act_upd_tot + np.sum(act_upd)
        print("Rec groups updated: " + str(act_upd_tot))


        return groups, cutoff, states, max_groups, np.mean(stepno)
    else:
        return cutoff


# TESTING

# This function assesses whether a pattern of interest is a stable state of a neuronal group.
def conv_achieved(abs_groups, des_pattern):
    is_match = []
    for i in range(len(abs_groups)):
        output, steps = updating(des_pattern, abs_groups[i])
        if np.array_equal(output, des_pattern) or np.array_equal(output, complement_pattern(des_pattern)):
            is_match.append(1)
        if i%10000 == 0:
            print(i)
    #print("Groups converged to pattern: " + str(int(np.sum(is_match))))
    return int(np.sum(is_match))


random.seed(0)
att_factor = 0.95
groups, con_groups = create_rec_rep(1000, 5, 10, 0.9, 0.5)
abs_groups, abs_con_groups = create_abs_rep(groups, con_groups, 250)
states = np.zeros(len(groups))