# Frozen Lakes
Frozen lake is a  environment from openai Gymnasiun

this is a mod from: 

[Arthur Juliani Article about Simple reinforment Learning with tensor flow Part 0](#https://awjuliani.medium.com/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)

The mod is an upgrade from OpenAiGym  to OpenAiGymnasium



## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. Clone the repository:
```bash
 git clone https://github.com/george-plab/OpenAI-FrozenLakeV1.git
```

2. optional but encouraged recommended install virtual enviroment:
```bash  
  pip install pipenv
  pipenv shell
```
3. Install requirement
```bash
  pipenv install -r requirements.txt
```
if you have any problem. instal one after the other one
```bash
  pipenv install transformers 
  pipenv torch 
  pipenv torchvision
  pipenv pandas 
```
## usage
### sumary

![Gif frozen lake](https://gymnasium.farama.org/_images/frozen_lake.gif)

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

[FrozenLake From Gymnasium](#https://gymnasium.farama.org/environments/toy_text/frozen_lake/)


## Changes from Gym to Gymnasium:

With Gym:
```bash

current_obs = env.reset()
   ....

next_obs, reward, Done, info = env.step(a)
    ...
current_obs = next_obs

```

With Gymnasyum:
```bash
obs = env.reset()
current_obs=obs[0]

    ...

next_obs, reward, terminated, truncated, info = env.step(a)
    ...
current_obs = next_obs

```



## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes.
4. Push your branch: `git push origin feature-name`.
5. Create a pull request.

## License
This project is licensed under the [MIT License](https://mit-license.org/).


