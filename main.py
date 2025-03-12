
"""
Created on Tue Feb 12 2025

Mod @author: VicAgent from: 
https://awjuliani.medium.com/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

This mod Update OpenAiGym-> OpenAiGynasyum 

This project is licensed under the [MIT License]
"""
"""
https://gymnasium.farama.org/environments/toy_text/frozen_lake/

"""
import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

#Initialize table with all zeros

Q = np.zeros((env.observation_space.n ,env.action_space.n))
print(env.observation_space.n)
print(env.action_space.n)


# Set learning parameters
lr = .7  
y = .9 # Factor de descuento
num_episodes = 4000
#create lists to contain total rewards and steps per episode

rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    obs = env.reset()
    current_obs=obs[0]
    rAll = 0
    done = False
    j = 0
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        Q[current_obs,0:env.action_space.n]+= np.random.randn(env.action_space.n)  * (1./(i+1))
       
        a = np.argmax(  Q[current_obs,0:env.action_space.n] )              
        #Get new state and reward from environment
        next_obs, reward, terminated, truncated, info = env.step(a)      
       
        #Update Q-Table with new knowledge
        Q[current_obs,a] = Q[current_obs,a] + lr*(reward + y*np.max(Q[next_obs,:]) - Q[current_obs,a])
        rAll += reward
        
        current_obs = next_obs
        done= terminated or truncated
        if done == True:
            break
   
    rList.append(rAll)

print("Score over time: {}".format(str(sum(rList)/num_episodes)))
print(f"Final Q-Table Values: {Q}")



