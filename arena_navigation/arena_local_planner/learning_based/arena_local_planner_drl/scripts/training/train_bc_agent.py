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

    # instantiate eval environment
    # take task_manager from first sim (currently evaluation only provided for single process)
    if ns_for_nodes:
        eval_env = DummyVecEnv([make_envs(args, ns_for_nodes, 0, params=params, PATHS=PATHS, train=False,)])
    else:
        eval_env = env

    # try to load most recent vec_normalize obj (contains statistics like moving avg)
    env, eval_env = load_vec_normalize(params, PATHS, env, eval_env)

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


    recording_path = os.path.join(RECORDINGS_DIR, args.recording + ".dictionary")
    print(recording_path)
    with open(recording_path, 'rb') as file:
        [epoch_obs, epoch_act] = pickle.load(file)
        print("number of episodes before trimming: " + str(len(epoch_obs)))

    transitions = []

    for i in range(1, len(epoch_obs)):

        episode_obs = np.array(epoch_obs[i])
        episode_act = np.array(epoch_act[i])[:,::2]
        #every second -> skip using the second linear velocity value -> This fixed the ValueError size (32, 3) != size (32, 2)

        if len(episode_obs) > 2 and len(episode_obs) < 200:
            
            transition = types.Trajectory(
                obs = episode_obs,
                acts = episode_act,
                infos = None,
                terminal = True
            )
            transitions.append(transition)

    print("number of episodes: " + str(len(transitions)))
    
    # start training
    start = time.time()

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


    # callback for evaluations after every epoch
    def eval_callback() -> None:
        is_success_buffer_2 = []

        # copied from sb3 code:
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
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        if len(is_success_buffer_2) > 0:
            success_rate = np.mean(is_success_buffer_2)
        else:
            success_rate = -1
        
        eval_list.append({
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_ep_length": mean_ep_length,
            "std_ep_length": std_ep_length,
            "success_rate": success_rate,
        })
        


    rng = np.random.default_rng(0)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=model.policy,
        demonstrations=transitions,
        rng=rng,
    )

    BC_epochs = 30

    eval_list = []

    bc_trainer.train(
        n_epochs=BC_epochs,
        on_epoch_end=eval_callback
    )

    # plot graphs after training
    x = [(i+1) for i in range(BC_epochs)]

    success_rates = [eval_result["success_rate"] for eval_result in eval_list]
    mean_rewards = [eval_result["mean_reward"] for eval_result in eval_list]
    mean_ep_lengths = [eval_result["mean_ep_length"] for eval_result in eval_list]

    plt.rcParams.update({'font.size': 11})

    fig = plt.figure(figsize=(12,4))

    ax1 = fig.add_subplot(131)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("mean episode length")
    ax2 = fig.add_subplot(132)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mean episode reward")
    ax3 = fig.add_subplot(133)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("success rate")

    ax1.plot(x, mean_ep_lengths)
    ax2.plot(x, mean_rewards)
    ax3.plot(x, success_rates)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PATHS["model"], "bc_epoch_eval.pdf"))
    plt.show()


    time_before_eval = time.time()-start
    print(f"Time passed before eval: {time_before_eval}s")

    # one evaluation at the end of training
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
