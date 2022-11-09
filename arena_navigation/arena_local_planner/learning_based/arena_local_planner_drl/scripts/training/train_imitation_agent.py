#!/usr/bin/env python
from typing import Any, Dict, List, Type, Union
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional

import gym
import numpy as np

import os, sys, rospy, time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy

from rl_agent.model.agent_factory import AgentFactory
from rl_agent.model.base_agent import BaseAgent
from tools.argsparser import parse_bc_training_args
from tools.custom_mlp_utils import *
from tools.train_agent_utils import *
from tools.staged_train_callback import InitiateNewTrainStage


# NEW IMPORTS:
from imitation.algorithms import bc
from imitation.data import types
from stable_baselines3.common.evaluation import evaluate_policy
import pickle
import numpy as np

import matplotlib.pyplot as plt

RECORDINGS_DIR = os.path.join(
    rospkg.RosPack().get_path("arena_local_planner_drl"), "recordings"
)

def main():
    args, _ = parse_bc_training_args()

    # in debug mode, we emulate multiprocessing on only one process
    # in order to be better able to locate bugs
    if args.debug:
        rospy.init_node("debug_node", disable_signals=False)

    # generate agent name and model specific paths
    #AGENT_NAME = get_agent_name(args)
    AGENT_NAME = "BC_" + get_agent_name(args) + args.recording
    PATHS = get_paths(AGENT_NAME, args)

    print("________ STARTING TRAINING WITH:  %s ________\n" % AGENT_NAME)

    # for training with start_arena_flatland.launch
    ns_for_nodes = "/single_env" not in rospy.get_param_names()

    # check if simulations are booted
    wait_for_nodes(with_ns=ns_for_nodes, n_envs=args.n_envs, timeout=5)

    # initialize hyperparameters (save to/ load from json)
    params = initialize_hyperparameters(
        PATHS=PATHS,
        load_target=args.load,
        config_name=args.config,
        n_envs=args.n_envs,
    )

    # instantiate train environment
    # when debug run on one process only
    if not args.debug and ns_for_nodes:
        env = SubprocVecEnv(
            [make_envs(args, ns_for_nodes, i, params=params, PATHS=PATHS) for i in range(args.n_envs)],
            start_method="fork",
        )
    else:
        env = DummyVecEnv([make_envs(args, ns_for_nodes, i, params=params, PATHS=PATHS) for i in range(args.n_envs)])

    # threshold settings for training curriculum
    # type can be either 'succ' or 'rew'
    """ trainstage_cb = InitiateNewTrainStage(
        n_envs=args.n_envs,
        treshhold_type="succ",
        upper_threshold=0.9,
        lower_threshold=0.7,
        task_mode=params["task_mode"],
        verbose=1,
    ) """

    # stop training on reward threshold callback
    """ stoptraining_cb = StopTrainingOnRewardThreshold(treshhold_type="succ", threshold=0.95, verbose=1) """

    # instantiate eval environment
    # take task_manager from first sim (currently evaluation only provided for single process)
    if ns_for_nodes:
        eval_env = DummyVecEnv([make_envs(args, ns_for_nodes, 0, params=params, PATHS=PATHS, train=False,)])
    else:
        eval_env = env

    # try to load most recent vec_normalize obj (contains statistics like moving avg)
    env, eval_env = load_vec_normalize(params, PATHS, env, eval_env)

    # evaluation settings
    # n_eval_episodes: number of episodes to evaluate agent on
    # eval_freq: evaluate the agent every eval_freq train timesteps
    """ eval_cb = EvalCallback(
        eval_env=eval_env,
        train_env=env,
        n_eval_episodes=100,
        eval_freq=15000,
        log_path=PATHS["eval"],
        best_model_save_path=PATHS["model"],
        deterministic=True,
        callback_on_eval_end=trainstage_cb,
        callback_on_new_best=stoptraining_cb,
    ) """
    
    # determine mode
    if args.custom_mlp:
        # custom mlp flag
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=args.net_arch, activation_fn=get_act_fn(args.act_fn)),
            gamma=params["gamma"],
            n_steps=params["n_steps"],
            ent_coef=params["ent_coef"],
            learning_rate=params["learning_rate"],
            vf_coef=params["vf_coef"],
            max_grad_norm=params["max_grad_norm"],
            gae_lambda=params["gae_lambda"],
            batch_size=params["m_batch_size"],
            n_epochs=params["n_epochs"],
            clip_range=params["clip_range"],
            tensorboard_log=PATHS["tb"],
            verbose=1,
        )
    elif args.agent is not None:
        agent: Union[Type[BaseAgent], Type[ActorCriticPolicy]] = AgentFactory.instantiate(args.agent)
        if isinstance(agent, BaseAgent):
            model = PPO(
                agent.type.value,
                env,
                policy_kwargs=agent.get_kwargs(),
                gamma=params["gamma"],
                n_steps=params["n_steps"],
                ent_coef=params["ent_coef"],
                learning_rate=params["learning_rate"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                gae_lambda=params["gae_lambda"],
                batch_size=params["m_batch_size"],
                n_epochs=params["n_epochs"],
                clip_range=params["clip_range"],
                tensorboard_log=PATHS.get("tb"),
                verbose=1,
            )
        elif issubclass(agent, ActorCriticPolicy):
            model = PPO(
                agent,
                env,
                gamma=params["gamma"],
                n_steps=params["n_steps"],
                ent_coef=params["ent_coef"],
                learning_rate=params["learning_rate"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                gae_lambda=params["gae_lambda"],
                batch_size=params["m_batch_size"],
                n_epochs=params["n_epochs"],
                clip_range=params["clip_range"],
                tensorboard_log=PATHS.get("tb"),
                verbose=1,
            )
        else:
            raise TypeError(
                f"Registered agent class {args.agent} is neither of type" "'BaseAgent' or 'ActorCriticPolicy'!"
            )
    else:
        # load flag
        if os.path.isfile(os.path.join(PATHS["model"], AGENT_NAME + ".zip")):
            model = PPO.load(os.path.join(PATHS["model"], AGENT_NAME), env)
        elif os.path.isfile(os.path.join(PATHS["model"], "best_model.zip")):
            model = PPO.load(os.path.join(PATHS["model"], "best_model"), env)
        update_hyperparam_model(model, PATHS, params, args.n_envs)





    #recording_path = "/home/liam/observations/" + args.recording + ".dictionary"
    recording_path = os.path.join(RECORDINGS_DIR, args.recording + ".dictionary")
    print(recording_path)
    #with open('/home/liam/observations/observations100_ROSNAV_Jackal_Map1.dictionary', 'rb') as file:
    #with open('/home/liam/observations/observations100_DWA_Jackal_EmptyMap1.dictionary', 'rb') as file:
    with open(recording_path, 'rb') as file:
        #epoch = pickle.load(file)
        [epoch_obs, epoch_act] = pickle.load(file)
        print("number of episodes before trimming: " + str(len(epoch_obs)))
        # TODO: load as correct data type? INSTEAD: throw away second linear part of action?
        # PROBLEM: robot is not holonomic -> second linear part should always be 0
        # !!! apparently is not the case when using DWA
        # when loading the "jackal" agent, this is the case


        # PROBLEM: start_training.launch has to be open and this script has to be executed twice,
        # or the MLP_Extractor has 3 inputs less for some reason

    transitions = []

    for i in range(1, len(epoch_obs)):
        #print("episode length: " + str(len(epoch[i])))

        episode_obs = np.array(epoch_obs[i])
        episode_act = np.array(epoch_act[i])[:,::2]
        #print(episode_obs.shape)

        if len(episode_obs) > 2 and len(episode_obs) < 200:
            #print("episode length: " + str(len(episode_obs)))
            #episode_act = episode_obs[1:,-3::2]
            """ if len(transitions) > 9:
                print("obs: " + str(episode_obs))
                print("act: " + str(episode_act)) """
            #every second -> skip using the second linear velocity value -> This fixed the ValueError size (32, 3) != size (32, 2)
            transition = types.Trajectory(
                obs = episode_obs,
                acts = episode_act,
                infos = None,
                terminal = True
            )
            transitions.append(transition)

    print("number of episodes: " + str(len(transitions)))

    """ act_array = obs_array[1:,-3:] """
    
    """ transitions = [types.Trajectory(
        obs = obs_array,
        acts = act_array,
        infos = None,
        terminal = False
    )] """

    # set num of timesteps to be generated
    """ n_timesteps = 40000000 if args.n is None else args.n """
    # start training
    start = time.time()
    """ try:
        model.learn(
            total_timesteps=n_timesteps,
            callback=eval_cb,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt..")
    # finally:
    # update the timesteps the model has trained in total
    # update_total_timesteps_json(n_timesteps, PATHS) """

    is_success_buffer = []

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

    def eval_callback() -> None:
        is_success_buffer_2 = []

        def log_success_callback2(locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
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
                    is_success_buffer_2.append(maybe_is_success)
                    #print("buffer length: " + str(len(is_success_buffer_2)))

        episode_rewards, episode_lengths = evaluate_policy(
            bc_trainer.policy,
            eval_env,
            n_eval_episodes=100,
            #render=self.render,
            deterministic=False,
            return_episode_rewards=True,
            warn=True,
            callback=log_success_callback2,
        )
        #print(f"is_success_buffer size: {len(is_success_buffer)}")
        #print(f"episode_rewards after training: {episode_rewards}")
        #print(f"episode_lengths after training: {episode_lengths}")
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        #print(f"Eval num_timesteps={100}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        #print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        if len(is_success_buffer_2) > 0:
            success_rate = np.mean(is_success_buffer_2)
            #print(f"Success rate: {100 * success_rate:.2f}%")
        else:
            success_rate = -1
        
        eval_list.append({
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_ep_length": mean_ep_length,
            "std_ep_length": std_ep_length,
            "success_rate": success_rate,
            #"sr_buffer_length": len(is_success_buffer_2)
        })
        


    rng = np.random.default_rng(0)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=model.policy,
        demonstrations=transitions,
        rng=rng,
    )

    # reward_before_training, _ = evaluate_policy(bc_trainer.policy, eval_env, 100)
    # print(f"Reward before training: {reward_before_training}")


    BC_epochs = 15

    eval_list = []

    bc_trainer.train(
        n_epochs=BC_epochs,
        #on_epoch_end=eval_callback
    )

    #reward_after_training, _ = evaluate_policy(bc_trainer.policy, eval_env, 100)
    # print(f"Reward before training: {reward_before_training}")
    #print(f"Reward after training: {reward_after_training}")

    """x = [i for i in range(BC_epochs)]
    #print("x: " + str(x))

    success_rates = [eval_result["success_rate"] for eval_result in eval_list]
    #print("success_rates" + str(success_rates))
    #plt.figure(1)
    plt.plot(x, success_rates)
    plt.xlabel("Epoch")
    plt.ylabel("Success Rate")
    plt.title("Success Rates")
    plt.savefig(os.path.join(PATHS["model"], "success_rates.png"))
    plt.show()


    mean_rewards = [eval_result["mean_reward"] for eval_result in eval_list]
    std_rewards = [eval_result["std_reward"] for eval_result in eval_list]
    #print("mean_rewards" + str(mean_rewards))
    #plt.figure(2)
    fig, ax = plt.subplots()
    ax.fill_between(x, np.add(mean_rewards, std_rewards), np.subtract(mean_rewards, std_rewards), alpha=.5, linewidth=0)
    ax.plot(x, mean_rewards)
    #plt.errorbar(x, mean_rewards, std_rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Reward")
    plt.title("Rewards")
    plt.savefig(os.path.join(PATHS["model"], "rewards.png"))
    plt.show()


    mean_ep_lengths = [eval_result["mean_ep_length"] for eval_result in eval_list]
    std_ep_lengths = [eval_result["std_ep_length"] for eval_result in eval_list]
    #print("mean_ep_lengths" + str(mean_ep_lengths))
    #plt.figure(3)
    fig, ax = plt.subplots()
    ax.fill_between(x, np.add(mean_ep_lengths, std_ep_lengths), np.subtract(mean_ep_lengths, std_ep_lengths), alpha=.5, linewidth=0)
    ax.plot(x, mean_ep_lengths)
    #plt.errorbar(x, mean_ep_lengths, std_ep_lengths)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Episode Length")
    plt.title("Episode Lengths")
    plt.savefig(os.path.join(PATHS["model"], "ep_lengths.png"))
    plt.show()


    i = 1
    for eval_results in eval_list:
        mean_reward = eval_results["mean_reward"]
        std_reward = eval_results["std_reward"]
        mean_ep_length = eval_results["mean_ep_length"]
        std_ep_length = eval_results["std_ep_length"]
        success_rate = eval_results["success_rate"]
        #sr_buffer_length = eval_results["sr_buffer_length"]
        print(f"After {i} epoch(s):")
        i += 1
        print(f"Eval num_timesteps={100}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        print(f"Success rate: {100 * success_rate:.2f}%")
        #print(f"Success buffer length: {sr_buffer_length:.2f}")"""


    time_before_eval = time.time()-start
    print(f"Time passed before eval: {time_before_eval}s")

    is_success_buffer = []
    episode_rewards, episode_lengths = evaluate_policy(
                bc_trainer.policy,
                eval_env,
                n_eval_episodes=100,
                #render=self.render,
                deterministic=False,
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

    model.policy = bc_trainer.policy

    os.makedirs(PATHS["model"], exist_ok=True)
    path = os.path.join(PATHS["model"], "best_model")
    model.save(path)
    print(f"Saved model to {path}")

    WORLD_PATH_PARAM = rospy.get_param("world_path")
    MAP = os.path.split(os.path.split(WORLD_PATH_PARAM)[0])[1]
    print(MAP)

    env.close()
    time_after_eval = time.time()-start
    print(f"Time passed after eval: {time_after_eval}s")

    info_dict = {
        "success_rate": str(success_rate),
        "mean_reward": str(mean_reward),
        "map": MAP,
        "recording": args.recording, 
        "time_before_eval": time_before_eval,
        "time_after_eval": time_after_eval,
        "BC_epochs": BC_epochs,
    }
    #os.makedirs(os.path.join(PATHS["model"], "info.json"))
    with open(os.path.join(PATHS["model"], "info.json"), "w") as info_file:
        json.dump(info_dict, info_file, indent=4)
    #os.makedirs(os.path.join(PATHS["model"], "info_SR"+str(success_rate)+"_MR"+str(mean_reward)+"_MAP"+MAP), exist_ok=True)
    print("created info file")

    print("Training script will be terminated")
    sys.exit()


if __name__ == "__main__":
    main()
