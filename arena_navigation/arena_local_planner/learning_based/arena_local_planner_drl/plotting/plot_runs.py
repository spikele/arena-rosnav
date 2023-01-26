import pickle
import numpy as np
import os, sys, rospy, time
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Type, Union

SAVE_PATH = os.path.dirname(os.path.realpath(__file__))
print(SAVE_PATH)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(BASE_PATH)

def get_data_from_npz(log_folder: str, agent_name: str):

    path = os.path.join(BASE_PATH, log_folder, "train_eval_log", agent_name, "evaluations.npz")

    file = np.load(path)

    if agent_name == "AGENT_24_2023_01_19__17_23":
        # manually add data from tensorboard log because it was missing in npz file
        missing_timesteps = np.array([60000, 120000, 180000, 240000, 300000, 360000, 420000, 480000, 540000, 600000, 660000, 720000, 780000, 840000, 900000, 960000, 1020000])
        missing_mean_episode_lengths = np.array([84.6, 75.25, 75.12, 83.22, 80.39, 69.51, 97.61, 96.16, 101.8, 94.33, 106.2, 89.09, 86.83, 97.85, 77.5, 96.69, 89.69])
        missing_mean_rewards = np.array([10.73, 9.657, 7.588, 9.282, 8.898, 7.271, 8.65, 8.59, 11.38, 10.91, 11.95, 11.03, 11.54, 13.24, 14.16, 11.93, 14.62])
        missing_success_rates = np.array([0.63, 0.6, 0.54, 0.6, 0.59, 0.56, 0.55, 0.55, 0.61, 0.61, 0.62, 0.61, 0.65, 0.67, 0.75, 0.64, 0.75])

        timesteps = np.concatenate((missing_timesteps, file["timesteps"]+1020000))
        mean_episode_lengths = np.concatenate((missing_mean_episode_lengths, np.mean(file["ep_lengths"], axis=1)))
        mean_rewards = np.concatenate((missing_mean_rewards, np.mean(file["results"], axis=1)))
        success_rates = np.concatenate((missing_success_rates, np.sum(file["successes"], axis=1)/100))
    else:
        timesteps = file["timesteps"]
        mean_episode_lengths = np.mean(file["ep_lengths"], axis=1)
        mean_rewards = np.mean(file["results"], axis=1)
        success_rates = np.sum(file["successes"], axis=1)/100

    return timesteps, mean_episode_lengths, mean_rewards, success_rates


def create_pdf(file_name, thesis_agents, thesis2_agents, labels, show_stages: bool = False, thesis_agents_load = None, thesis2_agents_load = None):
    all_data_to_plot = []

    if thesis_agents_load is not None:
        for i in range(len(thesis_agents)):
            name = thesis_agents[i]
            name_load = thesis_agents_load[i]
            timesteps, mean_episode_lengths, mean_rewards, success_rates = get_data_from_npz("training_logs_thesis", name)
            timesteps_load, mean_episode_lengths_load, mean_rewards_load, success_rates_load = get_data_from_npz("training_logs_thesis", name_load)
            all_data_to_plot.append((np.concatenate((timesteps, timesteps_load+timesteps[-1])), np.concatenate((mean_episode_lengths, mean_episode_lengths_load)), np.concatenate((mean_rewards, mean_rewards_load)), np.concatenate((success_rates, success_rates_load))))
    else:
        for name in thesis_agents:
            timesteps, mean_episode_lengths, mean_rewards, success_rates = get_data_from_npz("training_logs_thesis", name)
            all_data_to_plot.append((timesteps, mean_episode_lengths, mean_rewards, success_rates))

    if thesis2_agents_load is not None:
        for i in range(len(thesis2_agents)):
            name = thesis2_agents[i]
            name_load = thesis2_agents_load[i]
            timesteps, mean_episode_lengths, mean_rewards, success_rates = get_data_from_npz("training_logs_thesis2", name)
            timesteps_load, mean_episode_lengths_load, mean_rewards_load, success_rates_load = get_data_from_npz("training_logs_thesis2", name_load)
            all_data_to_plot.append((np.concatenate((timesteps, timesteps_load+timesteps[-1])), np.concatenate((mean_episode_lengths, mean_episode_lengths_load)), np.concatenate((mean_rewards, mean_rewards_load)), np.concatenate((success_rates, success_rates_load))))
    else:
        for name in thesis2_agents:
            timesteps, mean_episode_lengths, mean_rewards, success_rates = get_data_from_npz("training_logs_thesis2", name)
            all_data_to_plot.append((timesteps, mean_episode_lengths, mean_rewards, success_rates))


    num_agents = len(all_data_to_plot)
    print(num_agents)

    plt.rcParams.update({'font.size': 11})

    fig = plt.figure(figsize=(12,4.3))

    ax1 = fig.add_subplot(131)
    ax1.set_xlabel("time step")
    ax1.set_ylabel("mean episode length")
    ax2 = fig.add_subplot(132)
    ax2.set_xlabel("time step")
    ax2.set_ylabel("mean episode reward")
    ax3 = fig.add_subplot(133)
    ax3.set_xlabel("time step")
    ax3.set_ylabel("success rate")

    for i in range(num_agents):
        print(i)
        ax1.plot(all_data_to_plot[i][0], all_data_to_plot[i][1])
        ax2.plot(all_data_to_plot[i][0], all_data_to_plot[i][2])
        ax3.plot(all_data_to_plot[i][0], all_data_to_plot[i][3], label=labels[i])

    if show_stages:
        ax3.axhline(y = 0.9, color='black', linestyle='dotted')
        ax3.axhline(y = 0.5, color='black', linestyle='dotted')

    handles, labels = ax3.get_legend_handles_labels()
    ax2.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.0,-0.27,1,0.1), ncol=2, borderaxespad=0.0, mode="expand", handletextpad=0.4, handlelength=1.8),

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, file_name+".pdf"))
    plt.show()
"""
file_name = "BC_random_map1"
thesis_agents = ["AGENT_24_2023_01_18__03_48", "AGENT_24_2023_01_19__17_23"]
thesis2_agents = ["AGENT_24_2023_01_18__03_16", "AGENT_24_2023_01_19__16_20"]
labels = ["BC+PPO 1", "BC+PPO 3", "BC+PPO 2", "BC+PPO 4"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels)

file_name = "DAgger_random_map1"
thesis_agents = ["AGENT_24_2023_01_20__07_09"]
thesis2_agents = ["AGENT_24_2023_01_20__06_09"]
labels = ["DAgger+PPO 1", "DAgger+PPO 2"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels)

file_name = "random_both_maps"
thesis_agents = ["AGENT_24_2022_12_13__02_20", "AGENT_24_2022_11_10__23_00"]
thesis2_agents = ["AGENT_24_2022_11_09__15_59", "AGENT_24_2022_11_10__21_23"]
labels = ["Map_empty 1", "Map1 1", "Map_empty 2", "Map1 2"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels)


file_name = "staged_both_maps"
thesis_agents = ["AGENT_24_2022_11_19__14_08", "AGENT_24_2022_11_21__00_52"]
thesis2_agents = ["AGENT_24_2022_11_17__16_33", "AGENT_24_2022_11_19__13_06", "AGENT_24_2022_12_29__04_13"]
labels = ["Map_empty 1", "Map1 1", "Map_empty 2", "Map1 2", "Map1 3"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels, show_stages=True)

file_name = "small_staged_map_empty"
thesis_agents = ["AGENT_24_2022_11_23__00_23 copy"]
thesis2_agents = ["AGENT_24_2022_11_20__23_40 (copy 1)"]
thesis_agents_load = ["AGENT_24_2022_11_23__00_23"]
thesis2_agents_load = ["AGENT_24_2022_11_20__23_40"]
labels = ["Map_empty 1", "Map_empty 2"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels, show_stages=True, thesis_agents_load=thesis_agents_load, thesis2_agents_load=thesis2_agents_load)

file_name = "small_staged_map1"
thesis_agents = ["AGENT_24_2022_11_24__15_14 copy"]
thesis2_agents = ["AGENT_24_2022_11_23__21_22 (copy 1)"]
thesis_agents_load = ["AGENT_24_2022_11_24__15_14"]
thesis2_agents_load = ["AGENT_24_2022_11_23__21_22"]
labels = ["Map1 1", "Map1 2"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels, show_stages=True, thesis_agents_load=thesis_agents_load, thesis2_agents_load=thesis2_agents_load)

file_name = "SAC_staged_map1"
thesis_agents = ["AGENT_24_2022_12_01__17_52 copy"]
thesis2_agents = ["AGENT_24_2023_01_09__00_34"]
thesis_agents_load = ["AGENT_24_2022_12_01__17_52"]
thesis2_agents_load = None
labels = ["Map1 1", "Map1 2"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels, show_stages=True, thesis_agents_load=thesis_agents_load, thesis2_agents_load=thesis2_agents_load)

file_name = "SAC_small_staged_map_empty"
thesis_agents = ["AGENT_24_2022_11_27__19_50 copy"]
thesis2_agents = ["AGENT_24_2022_11_27__18_50 (copy 1)"]
thesis_agents_load = ["AGENT_24_2022_11_27__19_50"]
thesis2_agents_load = ["AGENT_24_2022_11_27__18_50"]
labels = ["Map_empty 1", "Map_empty 2"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels, show_stages=True, thesis_agents_load=thesis_agents_load, thesis2_agents_load=thesis2_agents_load)

file_name = "SAC_small_staged_map1"
thesis_agents = ["AGENT_24_2022_11_29__23_07 copy 2"]
thesis2_agents = ["AGENT_24_2022_11_29__21_58 (copy 1)"]
thesis_agents_load = ["AGENT_24_2022_11_29__23_07"]
thesis2_agents_load = ["AGENT_24_2022_11_29__21_58"]
labels = ["Map1 1", "Map1 2"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels, show_stages=True, thesis_agents_load=thesis_agents_load, thesis2_agents_load=thesis2_agents_load)"""
"""
file_name = "TQC_small_staged_map_empty"
thesis_agents = ["AGENT_24_2022_12_06__14_47 copy"]
thesis2_agents = ["AGENT_24_2022_12_03__18_25 (copy 1)"]
thesis_agents_load = ["AGENT_24_2022_12_06__14_47"]
thesis2_agents_load = ["AGENT_24_2022_12_03__18_25"]
labels = ["Map_empty 1", "Map_empty 2"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels, show_stages=True, thesis_agents_load=thesis_agents_load, thesis2_agents_load=thesis2_agents_load)

file_name = "TQC_small_staged_map1"
thesis_agents = ["AGENT_24_2022_12_04__14_07 copy"]
thesis2_agents = ["AGENT_24_2022_12_01__15_19 (copy 1)"]
thesis_agents_load = ["AGENT_24_2022_12_04__14_07"]
thesis2_agents_load = ["AGENT_24_2022_12_01__15_19"]
labels = ["Map1 1", "Map1 2"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels, show_stages=True, thesis_agents_load=thesis_agents_load, thesis2_agents_load=thesis2_agents_load)"""

"""file_name = "frame_stacking_small_staged_map1"
thesis_agents = ["AGENT_24_2022_12_09__19_28 copy"]
thesis2_agents = ["AGENT_24_2022_12_10__20_29 (copy 1)"]
thesis_agents_load = ["AGENT_24_2022_12_09__19_28"]
thesis2_agents_load = ["AGENT_24_2022_12_10__20_29"]
labels = ["Map1 1", "Map1 2"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels, show_stages=True, thesis_agents_load=thesis_agents_load, thesis2_agents_load=thesis2_agents_load)"""

file_name = "HER_staged3_map1"
thesis_agents = ["AGENT_24_2022_12_20__05_47"]
thesis2_agents = ["AGENT_24_2023_01_12__03_37"]
thesis_agents_load = None
thesis2_agents_load = None
labels = ["rule_05", "rule_her"]
create_pdf(file_name, thesis_agents, thesis2_agents, labels, show_stages=True, thesis_agents_load=thesis_agents_load, thesis2_agents_load=thesis2_agents_load)
