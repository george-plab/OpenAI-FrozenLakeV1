
"""
Created on Tue Feb 12 2025

@author: VicAgent 
"""
"""
Frozen lake environment from openai Gymnasiun. 

Map is grid of blocks, one start block and one end/goal block, safe frozen blocks or dangerous hole blocks
the agent/player moves until reach the goal
Rewards:
Reach goal: +1
Reach hole: 0
Reach frozen: 0

desc=None: Used to specify maps non-preloaded maps.
a custom map coulb be
desc=["SFFF", "FHFH", "FFFH", "HFFG"] 
S:start, H:hole, F:Frozen, G:Goal

map_name could be 4x4 or 8x8 blocks (preloaded maps)

"4x4":["SFFF","FHFH","FFFH", "HFFG"]

"8x8": ["SFFFFFFF","FFFFFFFF","FFFHFFFF","FFFFFHFF","FFFHFFFF","FHHFFFHF","FHFFHFHF","FFFHFFFG",]

If desc=None then map_name will be used. If both desc and map_name are None a random 8x8 map with 80% of locations frozen will be generated.

is_slippery=True: If true the player will move in intended direction with probability of 1/3
else will move in either perpendicular direction with equal probability of 1/3 in both directions.

https://gymnasium.farama.org/environments/toy_text/frozen_lake/

"""
import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

#Initialize table with all zeros

Q = np.zeros((env.observation_space.n ,env.action_space.n))
print(env.observation_space.n)
print(env.action_space.n)
#print(Q)
#print( Q[1,0:env.action_space.n] + np.random.randn(env.action_space.n)) 

# Set learning parameters
lr = .7  
y = .9 # Factor de descuento
num_episodes = 4000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    obs = env.reset()
    obs_index=obs[0]
    #print(obs)
    #print(obs_index)
    rAll = 0
    done = False
    j = 0
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        Q[obs_index,0:env.action_space.n]+= np.random.randn(env.action_space.n)  * (1./(i+1))
        #print(Q[obs_index,0:env.action_space.n])
        #a = np.argmax(  Q[obs, : ]  +  np.random.randn(1,env.action_space.n)  * (1./(i+1))  )
        a = np.argmax(  Q[obs_index,0:env.action_space.n] )
        #print(a)
        
        #Get new state and reward from environment
        next_obs, reward, terminated, truncated, info = env.step(a)
        #print(next_obs)
        #Update Q-Table with new knowledge
        Q[obs_index,a] = Q[obs_index,a] + lr*(reward + y*np.max(Q[next_obs,:]) - Q[obs_index,a])
        rAll += reward
        obs_index = next_obs
        done= terminated or truncated
        if done == True:
            break
    #jList.append(j)
    rList.append(rAll)

print("Score over time: {}".format(str(sum(rList)/num_episodes)))
print("Final Q-Table Values")
print(Q)


