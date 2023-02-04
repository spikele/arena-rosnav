# About This Fork

This fork extends the DRL training approach of Arena-Rosnav. The added methods/features include:

* Pre-training agents with either of the imitation learning algorithms Behavior Cloning (BC) and Dataset Aggregation (DAgger).
* Recording expert demonstrations to use for BC or DAgger.
* Training agents with different RL algorithms (SAC and TQC)
* Training agents with Hindsight Experience Replay
* Using frame-stacking for training

# Installation Instructions

Follow the usual installation instructions for Arena-Rosnav, but clone this fork and branch instead of the main version.

### Important:

A different version of Stable-Baselines3 than in the main version is needed. Install it by cloning [this](https://github.com/spikele/stable-baselines3) repo and installing it with pip:
```bash
git clone https://github.com/spikele/stable-baselines3
cd stable-baselines3
pip install -e .
```

Additionally, sb3-contrib and imitation need to be installed. Install them with:
```bash
pip install sb3-contrib==1.6.1
pip install imitation==0.3.2
```

# Imitation Learning

## Recording Expert Demonstrations

To record expert demonstrations, the main simulation of Arena-Rosnav is used. The following local planners (experts) are supported:
* DWA
* MPC
* TEB
* rosnav

To record, use the ROS launch file ```start_arena_flatland.launch``` with ```record_for_IL:=true```. An example call is:
```bash
roslaunch arena_bringup start_arena_flatland.launch map_file:="map1" local_planner:="dwa" model:="jackal" task_mode:="random_eval_one_map" scenario_file:="random_eval/random_indoor_obs20_rep100.json" record_for_IL:=true
```
In this example, the task mode random_eval_one_map is used to record the actions of DWA for 100 episodes on ```map1```. To use a DRL agent as expert, the following command could be used:
```bash
roslaunch arena_bringup start_arena_flatland.launch map_file:="map1" local_planner:="rosnav" agent_name:="AGENT_24_2022_11_23__00_23" model:="jackal" task_mode:="random_eval_one_map" scenario_file:="random_eval/random_indoor_obs20_rep100.json" record_for_IL:=true
```
To record on randomly generated maps, the following command could be used:
```bash
roslaunch arena_bringup start_arena_flatland.launch map_file:="random_map" local_planner:="dwa" model:="jackal" task_mode:="random_eval" scenario_file:="random_eval/random_indoor_obs20_rep100.json" record_for_IL:=true
```

## Training with BC

To train an agent with BC, first launch start_training.launch:
```bash
roslaunch arena_bringup start_training.launch train_mode:=true use_viz:=false task_mode:=random map_folder_name:="map1" num_envs:=1 model:="jackal"
```
Then start the training script with ```--il_alg bc```. An agent type, a config file and a recording of expert demonstrations should be given. Example:
```bash
roscd arena_local_planner_drl

python scripts/training/train_agent_thesis.py --agent AGENT_24 --n_envs 1 --config "jackal_random" --il_alg bc --recording recording__map1_ROSNAV_2023_01_19__16_27
```
The arguments ```--eval_il``` and ```--info_file``` can optionally be set to evaluate the agent after training with BC and to save the results from this evaluation in an info file:
```bash
python scripts/training/train_agent_thesis.py --agent AGENT_24 --n_envs 1 --config "jackal_random" --il_alg bc --recording recording__map1_ROSNAV_2023_01_19__16_27 --info_file --eval_il
```

## Training with DAgger

To train an agent with DAgger, first launch start_training.launch:
```bash
roslaunch arena_bringup start_training.launch train_mode:=true use_viz:=false task_mode:=random map_folder_name:="map1" num_envs:=1 model:="jackal"
``` 
Then start the training script with ```--il_alg dagger```. An agent type, a config file and an expert should be given. Example:
```bash
roscd arena_local_planner_drl

python scripts/training/train_agent_thesis.py --agent AGENT_24 --n_envs 1 --config "jackal_random" --il_alg dagger --expert AGENT_24_2023_01_18__03_48
```
Like for BC, the arguments ```--eval_il``` and ```--info_file``` can optionally be set to evaluate the agent after training with DAgger and to save the results from this evaluation in an info file:
```bash
python scripts/training/train_agent_thesis.py --agent AGENT_24 --n_envs 1  --config "jackal_random" --il_alg dagger --expert AGENT_24_2023_01_18__03_48 --info_file --eval_il
```
Also, a recording of expert demonstrations can optionally be given:
```bash
python scripts/training/train_agent_thesis.py --agent AGENT_24 --n_envs 1 --config "jackal_random" --il_alg dagger --recording recording__map1_ROSNAV_2023_01_19__16_27 --expert AGENT_24_2023_01_18__03_48
```

# DRL Training

## Training with PPO

To train an agent with PPO, first launch ```start_training.launch``` and then start the training script using a hyperparameters config file that has ```"algorithm": "ppo"```. The config file must define all the parameters required for the used algorithm. An Example for this is ```jackal_AIOS.json```.
```bash
roslaunch arena_bringup start_training.launch train_mode:=true use_viz:=false task_mode:=staged map_folder_name:="map1" num_envs:=4 model:="jackal"
``` 
Then start the training script. An agent type and a config file should be given. Example:
```bash
roscd arena_local_planner_drl

python scripts/training/train_agent_thesis.py --agent AGENT_24 --n_envs 4 --config "jackal_AIOS"
```

## Training with SAC and TQC

To train an agent with SAC or TQC, first launch ```start_training.launch``` and then start the training script using a hyperparameters config file that has ```"algorithm": "sac"``` or ```"algorithm": "tqc"```. The config file must define all the parameters required for the used algorithm. Examples for this are ```jackal_SAC_AIOS.json``` and ```jackal_TQC_AIOS.json```.

```bash
roslaunch arena_bringup start_training.launch train_mode:=true use_viz:=false task_mode:=staged map_folder_name:="map1" num_envs:=4 model:="jackal"
``` 
Then start the training script. An agent type and a config file should be given. Example:
```bash
roscd arena_local_planner_drl

python scripts/training/train_agent_thesis.py --agent AGENT_24 --n_envs 4 --config "jackal_SAC_AIOS"
```
Optionally, use ```--save_replay_buffer``` to disable saving the replay buffer whenever the model is saved. 
```bash
python scripts/training/train_agent_thesis.py --agent AGENT_24 --n_envs 4 --config "jackal_SAC_AIOS" --save_replay_buffer
```
If it is saved, the replay buffer will be loaded when the saved agent is loaded for continued training (with ```--load```). The replay buffer can also be used for training another agent with ```--load_replay_buffer```:
```bash
python scripts/training/train_agent_thesis.py --agent AGENT_24 --n_envs 4 --config "jackal_SAC_AIOS" --load_replay_buffer AGENT_24_2022_12_01__17_52
```

## Training with Frame-Stacking

For frame-stacking, first launch ```start_training.launch``` and then start the training script using a hyperparameters config file that has ```"use_frame_stacking": true```.
```bash
roslaunch arena_bringup start_training.launch train_mode:=true use_viz:=false task_mode:=staged map_folder_name:="map1" num_envs:=4 model:="jackal"
``` 
```bash
python scripts/training/train_agent_thesis.py --agent AGENT_24 --n_envs 4 --config "jackal_AIOS_fs"
```

## Hindsight Experience Replay

To train an agent with HER, first launch ```start_training.launch``` and then start the training script using a SAC or TQC hyperparameters config file that has ```"use_her": true```. It is only possible to use one parallel environment. Also, it is recommended to use the reward function ```rule_her``` (set in the hyperparameters config file). Example:
```bash
roslaunch arena_bringup start_training.launch train_mode:=true use_viz:=false task_mode:=staged map_folder_name:="map1" num_envs:=1 model:="jackal"
``` 
```bash
python scripts/training/train_agent_thesis.py --agent AGENT_24 --n_envs 1 --config "jackal_SAC_AIOS_HER"
```

## Other New Features

The argument ```--info_file``` can be used to enable that an info file is created. This contains the map used for training and the curriculum stages if using task mode staged.

The hyperparameters config file contains a parameter ```seed```. This determines the pseudorandom number generator seed given to the environment.