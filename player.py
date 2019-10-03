import numpy as np 
from simulator import Simulator, State
from policy import *
import ipdb
from copy import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(32)

# runs an episode with given policy in the given env, returns total reward for that episode
def episode(env, policy):
    state = env.reset()
    state, reward, done = env.check_after_init()
    while(not done):
        action = policy(state)
        state, reward, done = env.step(action)

    return reward

def state_transformation(state):
    return state.raw_sum + 30, state.trump_count(), state.opponent-1

def monte_carlo(env, policy, first_visit, num_episodes):
    # size of v = (rawsum, number of distinct trumps, dealer's hand)
    v = np.zeros((61,4,10), dtype=float)
    num_updates = np.zeros((61,4,10), dtype=float)
    
    for _ in tqdm(range(num_episodes)):
        # print("====================== NEW EPISODE ======================")
        states = []
        state = env.reset()
        state, reward, done = env.check_after_init()
        if done:
            # no actionable state encountered in this episode so no update
            continue
        states.append(copy(state))
        while(not done):
            action = policy(state)
            state, reward, done = env.step(action)
            states.append(copy(state))

        if states[-1] != None:
            raise Exception("last state in episode is actionable, CHECK")
        states = states[:-1]

        for s in states:
            if s.category=="BUST" or s.category=="SUM31":
                raise Exception("states within an episode are not actionable")

        # updating value function
        if first_visit:
            states = list(set(states))
        for state in states:
            transformed_state = state_transformation(state)
            v[transformed_state] += reward
            num_updates[transformed_state] += 1

    v/=num_episodes # not replacing nan with zeros to know which states were not updated

    return v

def k_step_TD(env, policy, k, alpha, num_episodes):
    # size of v = (rawsum, number of distinct trumps, dealer's hand)
    v = np.zeros((61,4,10), dtype=float)
    
    for _ in tqdm(range(num_episodes)):
        # print("====================== NEW EPISODE ======================")
        states = []
        state = env.reset()
        state, reward, done = env.check_after_init()
        if done:
            # no actionable state encountered in this episode so no update
            continue
        states.append(copy(state))
        
        # take k-1 steps
        for _ in range(k-1):
            action = policy(state)
            state, reward, done = env.step(action)
            if done:
                break
            states.append(copy(state))
        
        if not done:
            assert(len(states)==k), "number of states not correct"

        if(not done):
            while(True):
                action = policy(state)
                state, reward, done = env.step(action)
                if done:
                    break
                assert(reward==0), "reward is non-zero for intermediate states"
                # update S_t, remove from states list and add S_t+k to the states list
                initial_state = state_transformation(states[0])
                final_state = state_transformation(state)
                v[initial_state] += alpha * ( reward + v[final_state] - v[initial_state])
                states = states[1:] + [copy(state)]

        # TODO : states has both actionable and non-actionable states in it, DEBUG

        assert(states[-1]!=None), "states[-1] is None"

        # if states[-1] != None:
        #     raise Exception("last state in episode is actionable, CHECK")
        # states = states[:-1]

        for s in states:
            assert(s.category=="GENERAL"), "states within an episode are not actionable"
            # if s.category=="BUST" or s.category=="SUM31":
            #     raise Exception("states within an episode are not actionable")
            # else:
            #     s.print()
            
        # updating value of states after reaching end of episode
        for s in states:
            initial_state = state_transformation(s)
            v[initial_state] += alpha * ( reward - v[initial_state]) # last state is not actionable so its value is zero

    return v

env = Simulator()

# # average reward for a policy
# reward=0
# num_episodes = 100
# for i in range(num_episodes):
#     reward += episode(env, dealer_policy)
# print(reward/num_episodes)

# v = monte_carlo(env, dealer_policy, first_visit=True, num_episodes=100000)
for k in range(1,100):
    v = k_step_TD(env, dealer_policy, k=k, alpha=0.1, num_episodes=1000)

