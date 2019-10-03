import numpy as np 
from simulator import Simulator, State
from policy import *
import ipdb
from copy import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# np.random.seed(32)

def expand(states):
    for s in states:
        s[0].print()

# runs an episode with given policy in the given env, returns total reward for that episode
def episode(env, policy):
    state = env.reset()
    state, reward, done = env.check_after_init()
    while(not done):
        action = policy(state)
        state, reward, done = env.step(action)

    return reward

# returns average reward on running acc to greedy policy on a given q-function
def test(env, num_episodes, q, epsilon):
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()
        state, reward, done = env.check_after_init()
        while(not done):
            action = greedy(state, q)
            state, reward, done = env.step(action)
        total_reward += reward
    
    return total_reward/num_episodes

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

def k_step_sarsa(env, k, alpha, num_episodes, epsilon=None, epsilon_decay=False):
    # size of v = (actions, rawsum, number of distinct trumps, dealer's hand)
    q = np.zeros((61,4,10,2), dtype=float)
    # actions = {"HIT", "STICK"}

    for ep in tqdm(range(1, num_episodes+1)):
        # print("====================== NEW EPISODE ======================")
        # TODO : change decay rate suitably
        episode_epsilon = epsilon/(ep**0.1) if epsilon_decay else epsilon
        
        states = []
        state = env.reset()
        state, reward, done = env.check_after_init()
        if done:
            # no actionable state encountered in this episode so no update
            continue
        # states.append(copy((state,action)))

        action = epsilon_greedy(state, q, episode_epsilon)

        # take k-1 steps
        for _ in range(k-1):
            states.append((copy(state), action))
            state, reward, done = env.step(action)
            if done:
                break
            action = epsilon_greedy(state, q, episode_epsilon)

        if not done:
            assert(len(states)==k-1), "number of states not correct"

        if(not done):
            while(True):
                states.append((copy(state), action))
                state, reward, done = env.step(action)
                if done:
                    break
                assert(reward==0), "reward is non-zero for intermediate states"
                action = epsilon_greedy(state, q, episode_epsilon)

                # update S_t, remove from states list and add S_t+k to the states list
                initial_state = state_transformation(states[0][0])
                final_state = state_transformation(state)
                q[initial_state][0 if states[0][1]=="HIT" else 1] += alpha * ( reward + q[final_state][0 if action=="HIT" else 1] - q[initial_state][0 if states[0][1]=="HIT" else 1])
                states = states[1:] # + [copy((state, action))]

            assert(len(states)==k), ipdb.set_trace() # "number of states in window is not k"
            
        assert(states[-1]!=None), "states[-1] is None"

        # if states[-1] != None:
        #     raise Exception("last state in episode is actionable, CHECK")
        # states = states[:-1]

        for s in states:
            assert(s[0].category=="GENERAL"), "states within an episode are not actionable"
            
        # updating value of states after reaching end of episode
        for s in states:
            initial_state = state_transformation(s[0])
            q[initial_state][0 if s[1]=="HIT" else 1] += alpha * ( reward - q[initial_state][0 if s[1]=="HIT" else 1]) # last state is not actionable so its value is zero

    return q

def q_learning(env, alpha, num_episodes, epsilon=None, epsilon_decay=False):
    # size of v = (actions, rawsum, number of distinct trumps, dealer's hand)
    q = np.zeros((61,4,10,2), dtype=float)
    # actions = {"HIT", "STICK"}

    for ep in tqdm(range(1, num_episodes+1)):
        # print("====================== NEW EPISODE ======================")
        # TODO : change decay rate suitably
        episode_epsilon = epsilon/(ep**0.2) if epsilon_decay else epsilon
        
        # states = []
        state = env.reset()
        state, reward, done = env.check_after_init()
        if done:
            # no actionable state encountered in this episode so no update
            continue

        while(not done):
            prev_state = copy(state)
            action = epsilon_greedy(state, q, episode_epsilon)
            state, reward, done = env.step(action)

            if done:
                break
            assert(reward==0), "reward != 0 for actionable state"
            assert(state.category=="GENERAL"), "states within an episode are not actionable"

            # update q(s,a)
            initial_state = state_transformation(prev_state)
            final_state = state_transformation(state)
            q[initial_state][0 if action=="HIT" else 1] += alpha * (reward + max(q[final_state]) - q[initial_state][0 if action=="HIT" else 1])

        initial_state = state_transformation(prev_state)
        try:
            q[initial_state][0 if action=="HIT" else 1] += alpha * (reward - q[initial_state][0 if action=="HIT" else 1]) # last state is not actionable so its value is zero
        except:
            ipdb.set_trace()

    return q

def TD_lambda(env, alpha, lamda, num_episodes, epsilon=None, epsilon_decay=False):
    # size of v = (actions, rawsum, number of distinct trumps, dealer's hand)
    q = np.zeros((61,4,10,2), dtype=float)
    # actions = {"HIT", "STICK"}

    for ep in tqdm(range(1, num_episodes+1)):
        # print("====================== NEW EPISODE ======================")
        # TODO : change decay rate suitably
        episode_epsilon = epsilon/(ep**0.5) if epsilon_decay else epsilon
        
        states = []
        state = env.reset()
        state, reward, done = env.check_after_init()
        if done:
            # no actionable state encountered in this episode so no update
            continue

        while(not done):
            action = epsilon_greedy(state, q, episode_epsilon)
            states.append((copy(state), action))
            state, reward, done = env.step(action)

            if done:
                break
            assert(reward==0), "reward != 0 for actionable state"
            assert(state.category=="GENERAL"), "states within an episode are not actionable"

        episode_length = len(states)
        for i in range(len(states)):
            (s,a) = states[i]
            gt_lamda = 0
            for j,(s_,a_) in enumerate(states[i+1:]):
                final_state = state_transformation(s_)
                gt_lamda += (lamda**j) * q[final_state][0 if a_=="HIT" else 1]
            gt_lamda *= (1-lamda)
            gt_lamda += (lamda**(episode_length-i-1)) * reward

            initial_state = state_transformation(s)
            q[initial_state][0 if a=="HIT" else 1] += alpha * (gt_lamda - q[initial_state][0 if a=="HIT" else 1])

    return q

env = Simulator()

# # average reward for a policy
# reward=0
# num_episodes = 100
# for i in range(num_episodes):
#     reward += episode(env, dealer_policy)
# print(reward/num_episodes)


# # ===== MONTE CARLO =====
# v = monte_carlo(env, dealer_policy, first_visit=True, num_episodes=100000)

# # ===== K-STEP SARSA ===== 
# v = k_step_TD(env, dealer_policy, k=1, alpha=0.1, num_episodes=1000)

# for k in range(1,100):
#     v = k_step_TD(env, dealer_policy, k=k, alpha=0.1, num_episodes=1000)

# # ===== K-STEP SARSA WITH DECAYING EPSILON ===== 
# q = k_step_sarsa(env, k=5, alpha=0.1, num_episodes=100000, epsilon=0.1, epsilon_decay=True)

# # ===== Q-LEARNING ===== 
# q = q_learning(env, alpha=0.1, num_episodes=100000, epsilon=0.1, epsilon_decay=True)

# # ===== TD-LAMDA ===== 
q = TD_lambda(env, alpha=0.1, lamda=0.5, num_episodes=100000, epsilon=0.1, epsilon_decay=True)

# # ===== TESTING =====
reward = 0
for _ in tqdm(range(1000)):
    reward += (test(env, num_episodes=1000, q=q, epsilon=0.1))
print(reward/1000)
