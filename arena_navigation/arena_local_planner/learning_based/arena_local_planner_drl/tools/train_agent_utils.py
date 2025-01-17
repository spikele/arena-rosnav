from typing import Type, Union

import argparse
from datetime import datetime as dt
import gym
import json
import os, rospy
import rosnode
import rospkg
import time
import warnings

from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

import numpy as np

from sb3_contrib import TQC

from rl_agent.envs.flatland_gym_env import (
    FlatlandEnv, 
)
from rl_agent.envs.flatland_gym_env_her import FlatlandGoalEnv

from tools.constants import *


def initialize_hyperparameters(PATHS: dict, load_target: str, config_name: str = "default", n_envs: int = 1) -> dict:
    """
    Write hyperparameters to json file in case agent is new otherwise load existing hyperparameters

    :param PATHS: dictionary containing model specific paths
    :param load_target: unique agent name (when calling --load)
    :param config_name: name of the hyperparameter file in /configs/hyperparameters
    :param n_envs: number of envs
    """
    # when building new agent
    if load_target is None:
        hyperparams = load_hyperparameters_json(PATHS=PATHS,  from_scratch=True, config_name=config_name)
        hyperparams["robot"] = rospy.get_param("model", "not specified")
        hyperparams["agent_name"] = PATHS["model"].split("/")[-1]
    else:
        hyperparams = load_hyperparameters_json(PATHS=PATHS)

    if "actions_in_observationspace" not in hyperparams:
        hyperparams["actions_in_observationspace"] = False

    rospy.set_param(
        "actions_in_obs",
        hyperparams["actions_in_observationspace"],
    )

    import rl_agent.model.custom_policy
    import rl_agent.model.custom_sb3_policy

    # dynamically adapt n_steps according to batch size and n envs, or train_freq for off-policy algorithms
    if hyperparams["algorithm"] in ON_POLICY_ALGORITHMS:
        check_batch_size(n_envs, hyperparams["batch_size"], hyperparams["m_batch_size"])
        hyperparams["n_steps"] = int(hyperparams["batch_size"] / n_envs)
    if hyperparams["algorithm"] in OFF_POLICY_ALGORITHMS:
        hyperparams["train_freq"] = int(hyperparams["batch_size"] / n_envs)

    # update .json
    write_hyperparameters_json(hyperparams, PATHS)
    print_hyperparameters(hyperparams)
    return hyperparams


def load_hyperparameters_json(PATHS: dict, from_scratch: bool = False, config_name: str = "default") -> dict:
    """
    Load hyperparameters from model directory when loading - when training from scratch
    load from ../configs/hyperparameters

    :param PATHS: dictionary containing model specific paths
    :param from_scatch: if training from scratch
    :param config_name: file name of json file when training from scratch
    """
    if from_scratch:
        doc_location = os.path.join(PATHS.get("hyperparams"), config_name + ".json")
    else:
        doc_location = os.path.join(PATHS.get("model"), "hyperparameters.json")

    if os.path.isfile(doc_location):
        with open(doc_location, "r") as file:
            hyperparams = json.load(file)
        check_hyperparam_format(loaded_hyperparams=hyperparams, PATHS=PATHS)
        return hyperparams
    else:
        if from_scratch:
            raise FileNotFoundError("Found no '%s.json' in %s" % (config_name, PATHS.get("hyperparams")))
        else:
            raise FileNotFoundError("Found no 'hyperparameters.json' in %s" % PATHS.get("model"))


def check_hyperparam_format(loaded_hyperparams: dict, PATHS: dict) -> None:
    assert "algorithm" in loaded_hyperparams
    
    if loaded_hyperparams["algorithm"] == "sac":
        hyperparam_keys = HYPERPARAM_KEYS_SAC
    elif loaded_hyperparams["algorithm"] == "tqc":
        hyperparam_keys = HYPERPARAM_KEYS_TQC
    elif loaded_hyperparams["algorithm"] == "ppo":
        hyperparam_keys = HYPERPARAM_KEYS_PPO


    if set(hyperparam_keys.keys()) != set(loaded_hyperparams.keys()):
        missing_keys = set(hyperparam_keys.keys()).difference(
            set(loaded_hyperparams.keys())
        )
        redundant_keys = set(loaded_hyperparams.keys()).difference(
            set(hyperparam_keys.keys())
        )
        if missing_keys.difference(set(["actions_in_observationspace"])):
            raise AssertionError(
                f"unmatching keys, following keys missing: {missing_keys} \n"
                f"following keys unused: {redundant_keys}"
            )

        warnings.warn(
            "Couldn't find key 'actions_in_observationspace' hyperparameters json file!"
            "Add it manually, otherwise this value is set 'false' per default."
        )

    #assert loaded_hyperparams["algorithm"] == rl_alg
    assert loaded_hyperparams["algorithm"] in ON_POLICY_ALGORITHMS or loaded_hyperparams["algorithm"] in OFF_POLICY_ALGORITHMS
 
    if not isinstance(loaded_hyperparams["discrete_action_space"], bool):
        raise TypeError("Parameter 'discrete_action_space' not of type bool")
    if loaded_hyperparams["task_mode"] not in ["custom", "random", "staged"]:
        raise TypeError("Parameter 'task_mode' has unknown value")
    if (
        "actions_in_observationspace" in loaded_hyperparams
        and type(loaded_hyperparams["actions_in_observationspace"]) is not bool
    ):
        raise TypeError(
            "Parameter 'actions_in_observationspace' has to be a boolean!"
        )
    if loaded_hyperparams["algorithm"] in OFF_POLICY_ALGORITHMS:
        if not isinstance(loaded_hyperparams["use_her"], bool):
            raise TypeError("Parameter 'use_her' not of type bool")
    if not isinstance(loaded_hyperparams["use_frame_stacking"], bool):
        raise TypeError("Parameter 'use_frame_stacking' not of type bool")


def write_hyperparameters_json(hyperparams: dict, PATHS: dict) -> None:
    """
    Write hyperparameters.json to agent directory

    :param hyperparams: dict containing model specific hyperparameters
    :param PATHS: dictionary containing model specific paths
    """
    doc_location = os.path.join(PATHS.get("model"), "hyperparameters.json")

    with open(doc_location, "w", encoding="utf-8") as target:
        json.dump(hyperparams, target, ensure_ascii=False, indent=4)


def update_total_timesteps_json(timesteps: int, PATHS: dict) -> None:
    """
    Update total number of timesteps in json file

    :param hyperparams_obj(object, agent_hyperparams): object containing containing model specific hyperparameters
    :param PATHS: dictionary containing model specific paths
    """
    doc_location = os.path.join(PATHS.get("model"), "hyperparameters.json")
    hyperparams = load_hyperparameters_json(PATHS=PATHS)

    try:
        curr_timesteps = int(hyperparams["n_timesteps"]) + timesteps
        hyperparams["n_timesteps"] = curr_timesteps
    except Exception:
        raise Warning("Parameter 'total_timesteps' not found or not of type Integer in 'hyperparameter.json'!")
    else:
        with open(doc_location, "w", encoding="utf-8") as target:
            json.dump(hyperparams, target, ensure_ascii=False, indent=4)


def print_hyperparameters(hyperparams: dict) -> None:
    print("\n--------------------------------")
    print("         HYPERPARAMETERS         \n")
    for param, param_val in hyperparams.items():
        print("{:30s}{:<10s}".format((param + ":"), str(param_val)))
    print("--------------------------------\n\n")


def update_hyperparam_model(model: PPO, PATHS: dict, params: dict, n_envs: int = 1) -> None:
    """
    Updates parameter of loaded PPO agent when it was manually changed in the configs yaml.

    :param model(object, PPO): loaded PPO agent
    :param PATHS: program relevant paths
    :param params: dictionary containing loaded hyperparams
    :param n_envs: number of parallel environments
    """
    if params["algorithm"] == "ppo":
        if model.batch_size != params["m_batch_size"]:
            model.batch_size = params["m_batch_size"]
        if model.gamma != params["gamma"]:
            model.gamma = params["gamma"]
        if model.n_steps != params["n_steps"]:
            model.n_steps = params["n_steps"]
        if model.ent_coef != params["ent_coef"]:
            model.ent_coef = params["ent_coef"]
        if model.learning_rate != params["learning_rate"]:
            model.learning_rate = params["learning_rate"]
        if model.vf_coef != params["vf_coef"]:
            model.vf_coef = params["vf_coef"]
        if model.max_grad_norm != params["max_grad_norm"]:
            model.max_grad_norm = params["max_grad_norm"]
        if model.gae_lambda != params["gae_lambda"]:
            model.gae_lambda = params["gae_lambda"]
        if model.n_epochs != params["n_epochs"]:
            model.n_epochs = params["n_epochs"]
        """
        if model.clip_range != params['clip_range']:
            model.clip_range = params['clip_range']
        """
        if model.use_sde != params['use_sde']:
            model.use_sde = params['use_sde']
        if model.sde_sample_freq != params['sde_sample_freq']:
            model.sde_sample_freq = params['sde_sample_freq']
        if model.rollout_buffer.buffer_size != params["n_steps"]:
            model.rollout_buffer.buffer_size = params["n_steps"]
        if model.n_envs != n_envs:
            model.update_n_envs()

    if params["algorithm"] == "sac" or params["algorithm"] == "tqc":
        params_list = [
            "learning_rate",
            "buffer_size",
            "learning_starts",
            "tau",
            "gamma",
            "train_freq",
            "gradient_steps",
            "action_noise",
            "optimize_memory_usage",
            #"ent_coef", #should not be updated if it is 'auto'
            "target_update_interval",
            #"target_entropy", #should not be updated if it is 'auto'
            "use_sde",
            "sde_sample_freq",
            "use_sde_at_warmup",
        ]
        for param in params_list:
            if getattr(model, param) != params[param]:
                setattr(model, param, params[param])
        model._convert_train_freq()
        if model.batch_size != params["m_batch_size"]:
            model.batch_size = params["m_batch_size"]
        if model.n_envs != n_envs:
            model.n_envs = n_envs
            model.replay_buffer.n_envs = n_envs

    if params["algorithm"] == "tqc":
        if model.top_quantiles_to_drop_per_net != params["top_quantiles_to_drop_per_net"]:
            model.top_quantiles_to_drop_per_net = params["top_quantiles_to_drop_per_net"]

    
    if model.tensorboard_log != PATHS["tb"]:
        model.tensorboard_log = PATHS["tb"]


def check_batch_size(n_envs: int, batch_size: int, mn_batch_size: int) -> None:
    assert batch_size > mn_batch_size, f"Mini batch size {mn_batch_size} is bigger than batch size {batch_size}"

    assert (
        batch_size % mn_batch_size == 0
    ), f"Batch size {batch_size} isn't divisible by mini batch size {mn_batch_size}"

    assert batch_size % n_envs == 0, f"Batch size {batch_size} isn't divisible by n_envs {n_envs}"

    assert (
        batch_size % mn_batch_size == 0
    ), f"Batch size {batch_size} isn't divisible by mini batch size {mn_batch_size}"


def get_agent_name(args: argparse.Namespace) -> str:
    """Function to get agent name to save to/load from file system

    Example names:
    "MLP_B_64-64_P_32-32_V_32-32_relu_2021_01_07__10_32"
    "DRL_LOCAL_PLANNER_2021_01_08__7_14"

    :param args (argparse.Namespace): Object containing the program arguments
    """
    START_TIME = dt.now().strftime("%Y_%m_%d__%H_%M")

    if args.custom_mlp:
        return "MLP_B_" + args.body + "_P_" + args.pi + "_V_" + args.vf + "_" + args.act_fn + "_" + START_TIME
    if args.load is None:
        return args.agent + "_" + START_TIME
    return args.load


def get_paths(agent_name: str, args: argparse.Namespace) -> dict:
    """
    Function to generate agent specific paths

    :param agent_name: Precise agent name (as generated by get_agent_name())
    :param args (argparse.Namespace): Object containing the program arguments
    """
    dir = rospkg.RosPack().get_path("arena_local_planner_drl")
    robot_model = rospy.get_param("model")

    PATHS = {
        "model": os.path.join(dir, "agents", agent_name),
        "tb": os.path.join(dir, "training_logs", "tensorboard", agent_name),
        "eval": os.path.join(dir, "training_logs", "train_eval_log", agent_name),
        "robot_setting": os.path.join(
            rospkg.RosPack().get_path("simulator_setup"),
            "robot",
            f"{robot_model}.model.yaml",
        ),
        "hyperparams": os.path.join(dir, "configs", "hyperparameters"),
        "robot_as": os.path.join(dir, "configs", f"default_settings_{robot_model}.yaml"),
        "curriculum": os.path.join(dir, "configs", "training_curriculum_map1small.yaml"),
    }
    # check for mode
    if args.load is None:
        os.makedirs(PATHS["model"])
    elif not os.path.isfile(os.path.join(PATHS["model"], agent_name + ".zip")) and not os.path.isfile(
        os.path.join(PATHS["model"], "best_model.zip")
    ):
        raise FileNotFoundError(
            "Couldn't find model named %s.zip' or 'best_model.zip' in '%s'" % (agent_name, PATHS["model"])
        )
    # evaluation log enabled
    if args.eval_log:
        if not os.path.exists(PATHS["eval"]):
            os.makedirs(PATHS["eval"])
    else:
        PATHS["eval"] = None
    # tensorboard log enabled
    if args.tb:
        if not os.path.exists(PATHS["tb"]):
            os.makedirs(PATHS["tb"])
    else:
        PATHS["tb"] = None

    return PATHS


def get_expert_path(args: argparse.Namespace) -> dict:
    """
    Function to generate path of the expert that should be loaded

    :param args (argparse.Namespace): Object containing the program arguments
    """
    dir = rospkg.RosPack().get_path("arena_local_planner_drl")
    agent_name = args.expert

    return os.path.join(dir, "agents", agent_name)


def get_buffer_path(args: argparse.Namespace) -> dict:
    """
    Function to generate path of the replay buffer that should be loaded

    :param args (argparse.Namespace): Object containing the program arguments
    """
    dir = rospkg.RosPack().get_path("arena_local_planner_drl")
    agent_name = args.load_replay_buffer

    return os.path.join(dir, "agents", agent_name)


def make_envs(
    args: argparse.Namespace,
    with_ns: bool,
    rank: int,
    params: dict,
    # seed: int = 0,
    PATHS: dict = None,
    train: bool = True,
):
    """
    Utility function for multiprocessed env

    :param with_ns: (bool) if the system was initialized with namespaces
    :param rank: (int) index of the subprocess
    :param params: (dict) hyperparameters of agent to be trained
    # :param seed: (int) the inital seed for RNG
    :param PATHS: (dict) script relevant paths
    :param train: (bool) to differentiate between train and eval env
    :param args: (Namespace) program arguments
    :return: (Callable)
    """

    seed = params["seed"]

    def _init() -> Union[gym.Env, gym.Wrapper]:
        train_ns = f"sim_{rank+1}" if with_ns else ""
        eval_ns = f"eval_sim" if with_ns else ""

        if params["algorithm"] in OFF_POLICY_ALGORITHMS and params["use_her"]:
            if train:
                # train env
                env = FlatlandGoalEnv(
                    train_ns,
                    params["reward_fnc"],
                    params["discrete_action_space"],
                    goal_radius=params["goal_radius"],
                    max_steps_per_episode=params["train_max_steps_per_episode"],
                    debug=args.debug,
                    task_mode=params["task_mode"],
                    curr_stage=params["curr_stage"],
                    PATHS=PATHS,
                )
            else:
                # eval env
                env = Monitor(
                    FlatlandGoalEnv(
                        eval_ns,
                        params["reward_fnc"],
                        params["discrete_action_space"],
                        goal_radius=params["goal_radius"],
                        max_steps_per_episode=params["eval_max_steps_per_episode"],
                        train_mode=False,
                        debug=args.debug,
                        task_mode=params["task_mode"],
                        curr_stage=params["curr_stage"],
                        PATHS=PATHS,
                    ),
                    PATHS.get("eval"),
                    info_keywords=("done_reason", "is_success"),
                )
        else:
            if train:
                # train env
                env = FlatlandEnv(
                    train_ns,
                    params["reward_fnc"],
                    params["discrete_action_space"],
                    goal_radius=params["goal_radius"],
                    max_steps_per_episode=params["train_max_steps_per_episode"],
                    debug=args.debug,
                    task_mode=params["task_mode"],
                    curr_stage=params["curr_stage"],
                    PATHS=PATHS,
                )
            else:
                # eval env
                env = Monitor(
                    FlatlandEnv(
                        eval_ns,
                        params["reward_fnc"],
                        params["discrete_action_space"],
                        goal_radius=params["goal_radius"],
                        max_steps_per_episode=params["eval_max_steps_per_episode"],
                        train_mode=False,
                        debug=args.debug,
                        task_mode=params["task_mode"],
                        curr_stage=params["curr_stage"],
                        PATHS=PATHS,
                    ),
                    PATHS.get("eval"),
                    info_keywords=("done_reason", "is_success"),
                )

        
        
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def wait_for_nodes(with_ns: bool, n_envs: int, timeout: int = 30, nodes_per_ns: int = 3) -> None:
    """
    Checks for timeout seconds if all nodes to corresponding namespace are online.

    :param with_ns: (bool) if the system was initialized with namespaces
    :param n_envs: (int) number of virtual environments
    :param timeout: (int) seconds to wait for each ns
    :param nodes_per_ns: (int) usual number of nodes per ns
    """
    if with_ns:
        assert with_ns and n_envs >= 1, f"Illegal number of environments parsed: {n_envs}"
    else:
        assert not with_ns and n_envs == 1, f"Simulation setup isn't compatible with the given number of envs"

    for i in range(n_envs):
        for k in range(timeout):
            ns = "sim_" + str(i + 1) if with_ns else ""
            namespaces = rosnode.get_node_names(namespace=ns)

            if len(namespaces) >= nodes_per_ns:
                break

            warnings.warn(f"Check if all simulation parts of namespace '{ns}' are running properly")
            warnings.warn(f"Trying to connect again..")
            assert k < timeout - 1, f"Timeout while trying to connect to nodes of '{ns}'"

            time.sleep(1)


from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def load_vec_normalize(params: dict, PATHS: dict, env: VecEnv, eval_env: VecEnv):
    if params["normalize"]:
        load_path = os.path.join(PATHS["model"], "vec_normalize.pkl")
        if os.path.isfile(load_path):
            env = VecNormalize.load(load_path=load_path, venv=env)
            eval_env = VecNormalize.load(load_path=load_path, venv=eval_env)
            print("Succesfully loaded VecNormalize object from pickle file..")
        else:
            env = VecNormalize(
                env,
                training=True,
                norm_obs=True,
                norm_reward=False,
                clip_reward=15,
            )
            eval_env = VecNormalize(
                eval_env,
                training=True,
                norm_obs=True,
                norm_reward=False,
                clip_reward=15,
            )
    return env, eval_env


from rl_agent.model.agent_factory import AgentFactory
from rl_agent.model.base_agent import BaseAgent
from stable_baselines3.common.policies import ActorCriticPolicy
from tools.custom_mlp_utils import get_act_fn


def create_model(PATHS: dict, params: dict, args: argparse.Namespace, env: Union[gym.Env, VecEnv], agent_name: str, agent_type: str):
    if params["algorithm"] in OFF_POLICY_ALGORITHMS:
        if params["use_her"]:
            replay_buffer_class = HerReplayBuffer
            replay_buffer_kwargs = dict(
                n_sampled_goal=4,
                goal_selection_strategy='future',
                online_sampling=True,
                max_episode_length=params["train_max_steps_per_episode"],
            )
            agent_type += "_HER"
        else:
            replay_buffer_class = None
            replay_buffer_kwargs = None

    # determine mode
    if args.custom_mlp:
        # custom mlp flag
         
        if params["algorithm"] in OFF_POLICY_ALGORITHMS and params["use_her"]:
            # HER needs a modified features extractor, because it uses a dict observation space
            warnings.warn("HER is currently only supported for training with AGENT_24, training script may crash")

        if params["algorithm"] == "ppo":
            model = PPO(
                "MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=args.net_arch[0], activation_fn=get_act_fn(args.act_fn)),
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
                use_sde=params["use_sde"],
                sde_sample_freq=params["sde_sample_freq"],
                tensorboard_log=PATHS["tb"],
                verbose=1,
            )
        elif params["algorithm"] == "sac":
            model = SAC(
                "MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=args.net_arch[1], activation_fn=get_act_fn(args.act_fn)),
                learning_rate=params["learning_rate"],
                buffer_size=params["buffer_size"],
                learning_starts=params["learning_starts"],
                batch_size=params["m_batch_size"],
                tau=params["tau"],
                gamma=params["gamma"],
                train_freq=params["train_freq"],
                gradient_steps=params["gradient_steps"],
                action_noise=params["action_noise"],
                replay_buffer_class=replay_buffer_class,
                replay_buffer_kwargs=replay_buffer_kwargs,
                optimize_memory_usage=params["optimize_memory_usage"],
                ent_coef=params["ent_coef"],
                target_update_interval=params["target_update_interval"],
                target_entropy= params["target_entropy"],
                use_sde=params["use_sde"],
                sde_sample_freq=params["sde_sample_freq"],
                use_sde_at_warmup=params["use_sde_at_warmup"],
                tensorboard_log=PATHS.get("tb"),
                verbose=1,
            )
        elif params["algorithm"] == "tqc":
            model = TQC(
                "MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=args.net_arch[1], activation_fn=get_act_fn(args.act_fn)),
                learning_rate=params["learning_rate"],
                buffer_size=params["buffer_size"],
                learning_starts=params["learning_starts"],
                batch_size=params["m_batch_size"],
                tau=params["tau"],
                gamma=params["gamma"],
                train_freq=params["train_freq"],
                gradient_steps=params["gradient_steps"],
                action_noise=params["action_noise"],
                replay_buffer_class=replay_buffer_class,
                replay_buffer_kwargs=replay_buffer_kwargs,
                optimize_memory_usage=params["optimize_memory_usage"],
                ent_coef=params["ent_coef"],
                target_update_interval=params["target_update_interval"],
                target_entropy= params["target_entropy"],
                use_sde=params["use_sde"],
                sde_sample_freq=params["sde_sample_freq"],
                use_sde_at_warmup=params["use_sde_at_warmup"],
                top_quantiles_to_drop_per_net=params["top_quantiles_to_drop_per_net"],
                tensorboard_log=PATHS.get("tb"),
                verbose=1,
            )
    elif args.agent is not None:
        if args.agent != "AGENT_24":
            if params["use_her"]:
                # HER needs a modified features extractor, because it uses a dict observation space
                warnings.warn("HER is currently only supported for AGENT_24, training script may crash")
            if params["use_frame_stacking"]:
                # frame_stacking needs a modified features extractor if one of the custom ones in feature_extractors.py is used
                warnings.warn("frame stacking is currently only supported for AGENT_24, training script may crash")

        agent: Union[Type[BaseAgent], Type[ActorCriticPolicy]] = AgentFactory.instantiate(agent_type)
            
        if params["algorithm"] in OFF_POLICY_ALGORITHMS:
            # SAC and TQC do not allow shared layers in the net_arch, the shared layers will instead be added to both the q-function and the policy
            net_arch = agent.net_arch
            shared = net_arch[:-1]
            qf = shared + net_arch[-1]["vf"]
            pi = shared + net_arch[-1]["pi"]
            agent.net_arch = dict(qf=qf, pi=pi)

        if params["algorithm"] == "ppo":
            #agent: Union[Type[BaseAgent], Type[ActorCriticPolicy]] = AgentFactory.instantiate(agent_type)
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
                    use_sde=params["use_sde"],
                    sde_sample_freq=params["sde_sample_freq"],
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
                    use_sde=params["use_sde"],
                    sde_sample_freq=params["sde_sample_freq"],
                    tensorboard_log=PATHS.get("tb"),
                    verbose=1,
                )
            else:
                raise TypeError(
                    f"Registered agent class {args.agent} is neither of type" "'BaseAgent' or 'ActorCriticPolicy'!"
                )
        elif params["algorithm"] == "sac":
            #agent: Type[BaseAgent] = AgentFactory.instantiate(agent_type+"_off_policy")
            if isinstance(agent, BaseAgent):
                model = SAC(
                    agent.type.value,
                    env,
                    policy_kwargs=agent.get_kwargs(),
                    learning_rate=params["learning_rate"],
                    buffer_size=params["buffer_size"],
                    learning_starts=params["learning_starts"],
                    batch_size=params["m_batch_size"],
                    tau=params["tau"],
                    gamma=params["gamma"],
                    train_freq=params["train_freq"],
                    gradient_steps=params["gradient_steps"],
                    action_noise=params["action_noise"],
                    replay_buffer_class=replay_buffer_class,
                    replay_buffer_kwargs=replay_buffer_kwargs,
                    optimize_memory_usage=params["optimize_memory_usage"],
                    ent_coef=params["ent_coef"],
                    target_update_interval=params["target_update_interval"],
                    target_entropy= params["target_entropy"],
                    use_sde=params["use_sde"],
                    sde_sample_freq=params["sde_sample_freq"],
                    use_sde_at_warmup=params["use_sde_at_warmup"],
                    tensorboard_log=PATHS.get("tb"),
                    verbose=1,
                )
        elif params["algorithm"] == "tqc":
            #agent: Type[BaseAgent] = AgentFactory.instantiate(agent_type+"_off_policy")
            if isinstance(agent, BaseAgent):
                model = TQC(
                    agent.type.value,
                    env,
                    policy_kwargs=agent.get_kwargs(),
                    learning_rate=params["learning_rate"],
                    buffer_size=params["buffer_size"],
                    learning_starts=params["learning_starts"],
                    batch_size=params["m_batch_size"],
                    tau=params["tau"],
                    gamma=params["gamma"],
                    train_freq=params["train_freq"],
                    gradient_steps=params["gradient_steps"],
                    action_noise=params["action_noise"],
                    replay_buffer_class=replay_buffer_class,
                    replay_buffer_kwargs=replay_buffer_kwargs,
                    optimize_memory_usage=params["optimize_memory_usage"],
                    ent_coef=params["ent_coef"],
                    target_update_interval=params["target_update_interval"],
                    target_entropy= params["target_entropy"],
                    use_sde=params["use_sde"],
                    sde_sample_freq=params["sde_sample_freq"],
                    use_sde_at_warmup=params["use_sde_at_warmup"],
                    top_quantiles_to_drop_per_net=params["top_quantiles_to_drop_per_net"],
                    tensorboard_log=PATHS.get("tb"),
                    verbose=1,
                )
    else:
        # load flag
        if os.path.isfile(os.path.join(PATHS["model"], agent_name + ".zip")):
            path_to_load = os.path.join(PATHS["model"], agent_name)
        elif os.path.isfile(os.path.join(PATHS["model"], "best_model.zip")):
            path_to_load = os.path.join(PATHS["model"], "best_model")

        if params["algorithm"] == "ppo":
            model = PPO.load(path_to_load, env)
        elif params["algorithm"] == "sac":
            model = SAC.load(path_to_load, env)
        elif params["algorithm"] == "tqc":
            model = TQC.load(path_to_load, env)

        if params["algorithm"] in OFF_POLICY_ALGORITHMS:
            if os.path.isfile(os.path.join(PATHS["model"], "replay_buffer.pkl")):
                model.load_replay_buffer(os.path.join(PATHS["model"], "replay_buffer"))
                print("loaded replay buffer")

        update_hyperparam_model(model, PATHS, params, args.n_envs)

    print(model.policy)

    return model


import yaml


def save_info_dict(PATHS: dict, params: dict, info_dict: dict):
    WORLD_PATH_PARAM = rospy.get_param("world_path")
    MAP = os.path.split(os.path.split(WORLD_PATH_PARAM)[0])[1]
    print(MAP)

    info_dict["map"] = MAP

    if params["task_mode"] == "staged":
        # get the training curriculum stages
        # copied from tasks.py
        file_location = PATHS.get("curriculum")
        if os.path.isfile(file_location):
            with open(file_location, "r") as file:
                stages = yaml.load(file, Loader=yaml.FullLoader)
            assert isinstance(
                stages, dict
            ), "'training_curriculum.yaml' has wrong fromat! Has to encode dictionary!"
        else:
            raise FileNotFoundError(
                "Couldn't find 'training_curriculum.yaml' in %s "
                % PATHS.get("curriculum")
            )
        info_dict["stages"] = stages

    with open(os.path.join(PATHS["model"], "info.json"), "w") as info_file:
        json.dump(info_dict, info_file, indent=4)
    print("created info file")