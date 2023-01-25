# %%bash

# Install required system dependencies
!apt-get install -y xvfb x11-utils
# Install required python dependencies
!pip install gym[box2d]==0.17.* \
            pyvirtualdisplay==0.2.* \
            PyOpenGL==3.1.* \
            PyOpenGL-accelerate==3.1.*

# import pyvirtualdisplay
# _display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
# _ = _display.start()


# Install the gym and import required libraries 
import sys
!pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random

"""
Initialize a cartpole environment
"""
# Initialize 
env = gym.make("CartPole-v1")
print(env.action_space)
print(env.observation_space)
print(env.action_space.sample())

# Simulation for 100 steps
env.reset()

for i in range(100):
   #env.render()
   env.step(env.action_space.sample())
env.close()

# Simulation observation vector 
env.reset()

done = False
while not done:
   #env.render()
   obs, rew, done, info = env.step(env.action_space.sample())
   print(f"{obs} -> {rew}")
env.close()

# Get min and max values
print(env.observation_space.low)
print(env.observation_space.high) 

"""
State discretization
"""
# Function that will take the observation from the model and produce a tuple of 4 integer values
def discretize(x):
    return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))

# Another discretization method using bins
def create_bins(i,num):
    return np.arange(num+1)*(i[1]-i[0])/num+i[0]

print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))

ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
nbins = [20,20,10,10] # number of bins for each parameter
bins = [create_bins(ints[i],nbins[i]) for i in range(4)]

def discretize_bins(x):
    return tuple(np.digitize(x[i],bins[i]) for i in range(4))

# Simulation
env.reset()

done = False
while not done:
   #env.render()
   obs, rew, done, info = env.step(env.action_space.sample())
   #print(discretize_bins(obs))
   print(discretize(obs))
env.close()

"""
The Q-Table structure
"""
# Use the pair (state,action) as the dictionary key, and the value would correspond to Q-Table entry value
Q = {}
actions = (0,1)

def qvalues(state):
    return [Q.get((state,a),0) for a in actions]

"""
Q-Learning Algorithm
"""
# Set hyperparameters
alpha = 0.3
gamma = 0.9
epsilon = 0.90

# Collect all cumulative rewards at each simulation at rewards vector for further plotting
def probs(v,eps=1e-4):
    v = v-v.min()+eps
    v = v/v.sum()
    return v

Qmax = 0
cum_rewards = []
rewards = []
for epoch in range(100000):
    obs = env.reset()
    done = False
    cum_reward=0
    # == do the simulation ==
    while not done:
        s = discretize(obs)
        if random.random()<epsilon:
            # exploitation - chose the action according to Q-Table probabilities
            v = probs(np.array(qvalues(s)))
            a = random.choices(actions,weights=v)[0]
        else:
            # exploration - randomly chose the action
            a = np.random.randint(env.action_space.n)

        obs, rew, done, info = env.step(a)
        cum_reward+=rew
        ns = discretize(obs)
        Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
    cum_rewards.append(cum_reward)
    rewards.append(cum_reward)
    # == Periodically print results and calculate average reward ==
    if epoch%5000==0:
        print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
        if np.average(cum_rewards) > Qmax:
            Qmax = np.average(cum_rewards)
            Qbest = Q
        cum_rewards=[]

"""
Plotting Training Progress
"""
# Plotting rewards vector against the iteration number
plt.plot(rewards)

# Calculate the running average over a series of experiments to improve plot 
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))

"""
Varying Hyperparameters & Results
"""
# Trained model behavior simulation
obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   #env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()
