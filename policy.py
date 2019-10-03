import numpy as np
# from player import state_transformation

def state_transformation(state):
    return state.raw_sum + 30, state.trump_count(), state.opponent-1

def dealer_policy(state):
    if state.best_sum() < 25:
        return "HIT"
    else:
        return "STICK"

def always_hit(state):
    return "HIT"

def always_stick(state):
    return "STICK"

def greedy(state, q):
    state = state_transformation(state)
    if q[state][0] >= q[state][1]:
        return "HIT"
    else:
        return "STICK"

def epsilon_greedy(state, q, epsilon):
    greedy_action = greedy(state, q)
    non_greedy_action = "HIT" if greedy_action=="STICK" else "STICK"
    if np.random.rand(1)[0] < 1 - epsilon + (epsilon/2):
        return greedy_action
    else:
        return non_greedy_action
