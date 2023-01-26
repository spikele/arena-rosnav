#!/usr/bin/env python3

""" import bagpy
from bagpy import bagreader
import pandas as pd


b = bagreader('rosbags/dwa_map1_2022-05-21-15-23-00.bag')

# get the list of topics
# print(b.topic_table)

cmd_vel = pd.read_csv(b.message_by_topic("/cmd_vel"))
global_plan = pd.read_csv(b.message_by_topic("/move_base/DWAPlannerROS/global_plan"))
odom = pd.read_csv(b.message_by_topic("/odom"))
scan = pd.read_csv(b.message_by_topic("/scan"))
subgoal = pd.read_csv(b.message_by_topic("/subgoal"))

#print(cmd_vel)
print(global_plan) """
"""print(odom)
print(scan)
print(subgoal) """

""" print(cmd_vel.loc[0])
print("\n")
print(global_plan.loc[0])
print("\n")
print(odom.loc[0])
print("\n")
print(scan.loc[0])
print("\n") 
print(subgoal.loc[0]) """

import pickle
import numpy as np
from imitation.data import types

with open('observations/observations100_ROSNAV_Jackal_EmptyMap.dictionary', 'rb') as file:
    [epoch_obs, epoch_act] = pickle.load(file)
    print("number of episodes before trimming: " + str(len(epoch_obs)))
    print("number of episodes before trimming: " + str(len(epoch_act)))
    """ for i in range(0, len(obs_array)):
        #print(obs_array[i][-3:])
        if np.count_nonzero(obs_array[i,-3:]) == 0:
            print(" zero action at i=" + str(i)) """

#with open('/home/liam/observations/observationsTEST.dictionary', 'wb') as file:
#    pickle.dump([epoch_obs, epoch_act], file)
#    #pickle.dump(epoch_obs, file)
#     file.close()
#    print("saved observations")

transitions = []

for i in range(1, len(epoch_obs)):
    episode_obs = np.array(epoch_obs[i])
    episode_act = np.array(epoch_act[i])

    #print(episode_act)
    episode_act_2 = episode_act[:,::2]
    #print(episode_act)
    print(episode_obs.shape)
    print(episode_act.shape)

    if len(episode_obs) > 2:
        print("episode length: " + str(len(episode_obs)))
        #episode_act = episode_obs[1:,-3::2]
        #if len(transitions) > 9:
        #print("obs: " + str(episode_obs) + str(len(episode_obs)))
        #print("act: " + str(episode_act) + str(len(episode_act)))
        # NEW: every second -> skip using the second linear velocity value -> This fixed the ValueError size (32, 3) != size (32, 2)
        transition = types.Trajectory(
            obs = episode_obs,
            acts = episode_act_2,
            infos = None,
            terminal = True
        )
        transitions.append(transition)

        #reward. info = get_reward(episode_act[-1], episode_obs[-1])

print("number of episodes: " + str(len(transitions)))