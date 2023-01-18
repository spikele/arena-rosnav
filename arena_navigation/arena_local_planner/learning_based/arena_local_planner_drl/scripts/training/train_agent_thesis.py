#!/usr/bin/env python
from typing import Type, Union

import os, sys, rospy, time, tempfile

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy

from rl_agent.model.agent_factory import AgentFactory
from rl_agent.model.base_agent import BaseAgent
from tools.argsparser import parse_training_args_thesis
from tools.custom_mlp_utils import *
from tools.train_agent_utils import *
from tools.staged_train_callback import InitiateNewTrainStage
from tools.imitation_utils import *

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer


def main():
    args, _ = parse_training_args_thesis()

    # in debug mode, we emulate multiprocessing on only one process
    # in order to be better able to locate bugs
    if args.debug:
        rospy.init_node("debug_node", disable_signals=False)

    # generate agent name and model specific paths
    AGENT_NAME = get_agent_name(args)
    PATHS = get_paths(AGENT_NAME, args)

    agent_type = args.agent

    print("________ STARTING TRAINING WITH:  %s ________\n" % AGENT_NAME)

    # for training with start_arena_flatland.launch
    ns_for_nodes = "/single_env" not in rospy.get_param_names()

    # check if simulations are booted
    wait_for_nodes(with_ns=ns_for_nodes, n_envs=args.n_envs, timeout=5)

    # initialize hyperparameters (save to/ load from json)
    params = initialize_hyperparameters_thesis(
        PATHS=PATHS,
        load_target=args.load,
        config_name=args.config,
        n_envs=args.n_envs,
        #rl_alg=args.rl_alg,
    )

    # instantiate train environment
    # when debug run on one process only
    if not args.debug and ns_for_nodes:
        env = SubprocVecEnv(
            [make_envs_thesis(args, ns_for_nodes, i, params=params, PATHS=PATHS) for i in range(args.n_envs)],
            start_method="fork",
        )
    else:
        env = DummyVecEnv([make_envs_thesis(args, ns_for_nodes, i, params=params, PATHS=PATHS) for i in range(args.n_envs)])

    # threshold settings for training curriculum
    # type can be either 'succ' or 'rew'
    trainstage_cb = InitiateNewTrainStage(
        n_envs=args.n_envs,
        treshhold_type="succ",
        upper_threshold=0.9,
        lower_threshold=0.5,
        task_mode=params["task_mode"],
        verbose=1,
    )

    # stop training on reward threshold callback
    stoptraining_cb = StopTrainingOnRewardThreshold(
        treshhold_type="succ", threshold=0.95, verbose=1
    )

    # instantiate eval environment
    # take task_manager from first sim (currently evaluation only provided for single process)
    if ns_for_nodes:
        eval_env = DummyVecEnv(
            [
                make_envs_thesis(
                    args,
                    ns_for_nodes,
                    0,
                    params=params,
                    PATHS=PATHS,
                    train=False,
                )
            ]
        )
    else:
        eval_env = env

    # try to load most recent vec_normalize obj (contains statistics like moving avg)
    env, eval_env = load_vec_normalize(params, PATHS, env, eval_env)

    # use a frame-stacking wrapper for the environments if using framestacking
    if params["use_frame_stacking"]:
        env = VecFrameStack(env, 4)
        eval_env = VecFrameStack(eval_env, 4)
        if agent_type is not None:
            agent_type += "_frame_stacking"

    # evaluation settings
    # n_eval_episodes: number of episodes to evaluate agent on
    # eval_freq: evaluate the agent every eval_freq train timesteps
    eval_cb = EvalCallback(
        eval_env=eval_env,
        train_env=env,
        n_eval_episodes=100,
        eval_freq=15000,
        log_path=PATHS["eval"],
        best_model_save_path=PATHS["model"],
        save_replay_buffer=args.save_replay_buffer,
        deterministic=True,
        callback_on_eval_end=trainstage_cb,
        #callback_on_new_best=stoptraining_cb,
    )

    '''# determine mode
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

    print(model.policy)'''


    info_dict = {}

    model = create_model(PATHS=PATHS, params=params, args=args, env=env, agent_name=AGENT_NAME, agent_type=agent_type)

    if args.load_replay_buffer is not None:
        if params["algorithm"] in OFF_POLICY_ALGORITHMS:
            BUFFER_PATH = get_buffer_path(args)
            print("loading replay buffer")
            model.load_replay_buffer(os.path.join(BUFFER_PATH, "replay_buffer"))
            print("finished loading replay buffer")
            info_dict["loaded_replay_buffer"] = BUFFER_PATH
        else:
            warnings.warn("argument load_replay_buffer is ignored for on-policy algorithms")

    
    

    if args.il_alg in ["bc", "dagger"]:
        il_start = time.time()

        transitions = load_recording(args=args)

        rng = np.random.default_rng(params["seed"])

        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            policy=model.policy,
            demonstrations=transitions,
            rng=rng,
        )

        if args.il_alg == "bc":
            BC_epochs = 15

            #eval_list = []

            bc_trainer.train(
                n_epochs=BC_epochs,
                #on_epoch_end=eval_callback
            )

            model.policy = bc_trainer.policy

        elif args.il_alg == "dagger":
            expert_model = load_expert(PATHS=PATHS, args=args)

            DAgger_timesteps = 2000
            BC_epochs = 4

            with tempfile.TemporaryDirectory(prefix="dagger_") as tmpdir:
                dagger_trainer = SimpleDAggerTrainer(
                    venv=env,
                    scratch_dir=tmpdir,
                    expert_policy=expert_model,
                    bc_trainer=bc_trainer,
                    rng=rng,
                )
                dagger_trainer.train(total_timesteps=DAgger_timesteps, bc_train_kwargs={"n_epochs": BC_epochs})

            model.policy = dagger_trainer.policy

            info_dict["DAgger_timesteps"] = DAgger_timesteps

        il_time_before_eval = time.time()-il_start

        save_model(model=model, PATHS=PATHS)

        info_dict["recording"] = args.recording
        info_dict["BC_epochs"] = BC_epochs
        info_dict["time_before_eval"] = il_time_before_eval

        if args.eval_il:
            mean_reward, std_reward, mean_ep_length, std_ep_length, success_rate = evaluate_il(model, eval_env)

            il_time_after_eval = time.time()-il_start

            info_dict["time_after_eval"] = il_time_after_eval
            info_dict["mean_reward"] = str(mean_reward)
            info_dict["std_reward"] = str(std_reward)
            info_dict["mean_ep_length"] = str(mean_ep_length)
            info_dict["std_ep_length"] = str(std_ep_length)
            info_dict["success_rate"] = str(success_rate)
        
        
    if args.info_file:
        save_info_dict(PATHS=PATHS, params=params, info_dict=info_dict)

    if args.il_alg == "":
        # set num of timesteps to be generated
        n_timesteps = 2000000 if args.n is None else args.n
        # start training
        start = time.time()
        try:
            model.learn(
                total_timesteps=n_timesteps,
                callback=eval_cb,
                reset_num_timesteps=True,
            )
        except KeyboardInterrupt:
            print("KeyboardInterrupt..")

        print(f"Time passed: {time.time()-start}s")
        # finally:
        # update the timesteps the model has trained in total
        #update_total_timesteps_json(n_timesteps, PATHS)

    model.env.close()
    print("Training script will be terminated")
    sys.exit()


if __name__ == "__main__":
    main()
