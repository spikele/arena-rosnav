import pickle
import numpy as np
import os, rospy, rospkg
from typing import Type, Union, Any, Dict
import gym
import argparse
import json

from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC
from imitation.algorithms import bc
from imitation.data import types

from tools.train_agent_utils import update_hyperparam_model_thesis


def load_recording(args: argparse.Namespace):
    if args.recording != "":
        RECORDINGS_DIR = os.path.join(rospkg.RosPack().get_path("arena_local_planner_drl"), "recordings")
        recording_path = os.path.join(RECORDINGS_DIR, args.recording + ".dictionary")
        print(recording_path)
        with open(recording_path, 'rb') as file:
            [epoch_obs, epoch_act] = pickle.load(file)

        transitions = []

        for i in range(1, len(epoch_obs)):
            episode_obs = np.array(epoch_obs[i])
            episode_act = np.array(epoch_act[i])[:,::2]

            transition = types.Trajectory(
                obs = episode_obs,
                acts = episode_act,
                infos = None,
                terminal = True
            )
            transitions.append(transition)

        print("number of expert demonstration episodes: " + str(len(transitions)))

    else:
        transitions = None

    return transitions
    

def get_expert_path(args: argparse.Namespace) -> dict:
    """
    Function to generate path of the expert that should be loaded

    :param args (argparse.Namespace): Object containing the program arguments
    """
    dir = rospkg.RosPack().get_path("arena_local_planner_drl")
    agent_name = args.expert

    return os.path.join(dir, "agents_thesis", agent_name)


def save_model(model: BaseAlgorithm, PATHS: dict):
    os.makedirs(PATHS["model"], exist_ok=True)
    path = os.path.join(PATHS["model"], "best_model")
    model.save(path)
    print(f"Saved model to {path}")


def load_expert(PATHS: dict, args: argparse.Namespace, env: Union[gym.Env, VecEnv]):
    EXPERT_PATH = get_expert_path(args)

    # load hyperparams from the json at EXPERT_PATH
    doc_location = os.path.join(EXPERT_PATH, "hyperparameters.json")
    if os.path.isfile(doc_location):
        with open(doc_location, "r") as file:
            params = json.load(file)
    else:
        raise FileNotFoundError("Found no 'hyperparameters.json' in %s" % EXPERT_PATH)

    if os.path.isfile(os.path.join(EXPERT_PATH, args.expert + ".zip")):
        path_to_load = os.path.join(EXPERT_PATH, args.expert)
    elif os.path.isfile(os.path.join(EXPERT_PATH, "best_model.zip")):
        path_to_load = os.path.join(EXPERT_PATH, "best_model")

    if params["algorithm"] == "ppo":
        model = PPO.load(path_to_load, env)
    elif params["algorithm"] == "sac":
        model = SAC.load(path_to_load, env)
    elif params["algorithm"] == "tqc":
        model = TQC.load(path_to_load, env)

    update_hyperparam_model_thesis(model, PATHS, params, args.n_envs)

    return model


def evaluate_il(model: BaseAlgorithm, env: Union[gym.Env, VecEnv]):
    # copied from sb3 code:
    def log_success_callback(locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """     
        if locals_["done"]:
            maybe_is_success = locals_["info"]["is_success"]
            #print("maybe is success: " + str(maybe_is_success))
            if maybe_is_success is not None:
                is_success_buffer.append(maybe_is_success)
                #print("buffer length: " + str(len(is_success_buffer)))

    is_success_buffer = []
    episode_rewards, episode_lengths = evaluate_policy(
                model,
                env,
                n_eval_episodes=100,
                deterministic=True,
                return_episode_rewards=True,
                warn=True,
                callback=log_success_callback,
            )
    print(f"is_success_buffer size: {len(is_success_buffer)}")
    print(f"episode_rewards after training: {episode_rewards}")
    print(f"episode_lengths after training: {episode_lengths}")
    mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
    mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
    print(f"Eval num_timesteps={100}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
    if len(is_success_buffer) > 0:
        success_rate = np.mean(is_success_buffer)
        print(f"Success rate: {100 * success_rate:.2f}%")
    else:
        success_rate = 0
    
    return mean_reward, std_reward, mean_ep_length, std_ep_length, success_rate