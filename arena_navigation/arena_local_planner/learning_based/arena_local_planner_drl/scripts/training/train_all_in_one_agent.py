import os
import random
import sys
import time
import warnings
import json
from multiprocessing import Process

import rosnode
import rospkg
from rl_agent.envs.all_in_one_flatland_gym_env import AllInOneEnv
from rl_agent.envs.all_in_one_models.drl.drl_agent import setup_and_start_drl_server
from scripts.custom_policy import policy_kwargs_agent_13
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from tools import argsparser
from tools.train_agent_utils import initialize_hyperparameters


def make_all_in_one_envs(rank: int, paths: dict, params: dict, train: bool = True, seed: int = 0,
                         drl_server_url: [str] = None, num_envs: int = 0, eval_episodes: int = 0):
    def _init():
        if drl_server_url is not None:
            server_numb = rank // (num_envs // len(drl_server_url))
            drl_server_url_ind = drl_server_url[server_numb]
        else:
            drl_server_url_ind = None

        if train:
            all_in_one_env = AllInOneEnv(f"sim_{rank + 1}", paths['robot_setting'], paths['robot_as'],
                                         params['reward_fnc'],
                                         goal_radius=params['goal_radius'], paths=paths,
                                         max_steps_per_episode=params['train_max_steps_per_episode'],
                                         drl_server=drl_server_url_ind)
        else:
            seed = random.randint(1,1000)
            all_in_one_env = Monitor(
                AllInOneEnv("eval_sim", paths['robot_setting'], paths['robot_as'], params['reward_fnc'],
                            goal_radius=params['goal_radius'],
                            paths=paths, train_mode=False, max_steps_per_episode=params['eval_max_steps_per_episode'],
                            drl_server=drl_server_url_ind, evaluation=True, seed=seed,
                            evaluation_episodes=eval_episodes),
                paths.get('eval'),
                info_keywords=("done_reason", "is_success"))
        all_in_one_env.seed(seed + rank)
        return all_in_one_env

    set_random_seed(seed)
    return _init


def wait_for_nodes(n_envs: int, timeout: int = 30, nodes_per_ns: int = 3):
    """
    Checks for timeout seconds if all nodes to corresponding namespace are online.

    :param n_envs: (int) number of virtual environments
    :param timeout: (int) seconds to wait for each ns
    :param nodes_per_ns: (int) usual number of nodes per ns
    """
    for i in range(n_envs):
        for k in range(timeout):
            ns = rosnode.get_node_names(namespace='sim_' + str(i + 1))

            if len(ns) < nodes_per_ns:
                warnings.warn(
                    f"Check if all simulation parts of namespace '{'/sim_' + str(i + 1)}' are running properly")
                warnings.warn(f"Trying to connect again..")
            else:
                break

            assert (k < timeout - 1
                    ), f"Timeout while trying to connect to nodes of '{'/sim_' + str(i + 1)}'"

            time.sleep(1)


def get_paths(agent_version: str, args, all_in_one_config: str = "all_in_one_default.json") -> dict:
    """
    Function to generate agent specific paths

    :param agent_name: Precise agent name (as generated by get_agent_name())
    :param args (argparse.Namespace): Object containing the program arguments
    """
    dir = rospkg.RosPack().get_path('arena_local_planner_drl')

    paths = {
        'model':
            os.path.join(
                dir, 'agents', agent_version),
        'tb':
            os.path.join(
                dir, 'training_logs', 'tensorboard', agent_version),
        'eval':
            os.path.join(
                dir, 'training_logs', 'train_eval_log', agent_version),
        'robot_setting':
            os.path.join(
                rospkg.RosPack().get_path('simulator_setup'),
                'robot', 'myrobot' + '.model.yaml'),
        'hyperparams':
            os.path.join(
                dir, 'configs', 'hyperparameters'),
        'robot_as':
            os.path.join(
                dir, 'configs', 'default_settings.yaml'),
        'curriculum':
            os.path.join(
                dir, 'configs', 'training_curriculum_map1small.yaml'),
        'all_in_one_parameters':
            os.path.join(dir, 'configs', 'all_in_one_hyperparameters', all_in_one_config),
        'drl_agents':
            os.path.join(
                dir, 'agents')
    }

    if not os.path.exists(paths['model']):
        os.makedirs(paths['model'])

    if args.tb:
        if not os.path.exists(paths['tb']):
            os.makedirs(paths['tb'])
    else:
        paths['tb'] = None
    return paths


def set_up_drl_server(all_in_one_config_path: str):
    # setup drl server if required
    config_path = paths['all_in_one_parameters']
    models = []
    with open(config_path, 'r') as model_json:
        config_data = json.load(model_json)

    assert config_data is not None, "Error: All in one parameter file cannot be found!"

    config_model = config_data['models']

    if 'drl' in config_model and 'use_drl_servers' in config_model:
        socket_url_server_array = []
        socket_url_client_array = []
        server_base_url = "tcp://*:555"
        client_base_url = "tcp://localhost:555"
        count = 5
        for i in range(config_model['config_model']):
            socket_url_server_array.append(server_base_url + str(count))
            socket_url_client_array.append(client_base_url + str(count))
            count += 1
        Process(target=setup_and_start_drl_server, args=(socket_url_server_array, paths)).start()
        return socket_url_client_array
    else:
        return None


if __name__ == '__main__':

    # make unique agent version description based on @version
    version = "all_in_one_agents_teb_rlca_rule03_policy13"
    all_in_one_config = "rlca_teb_only.json"
    eval_episodes = 40

    add_version = 0
    main_dir = rospkg.RosPack().get_path('arena_local_planner_drl')
    while os.path.exists(os.path.join(main_dir, 'agents', version + str(add_version))):
        add_version += 1
    version = version + ".v" + str(add_version)

    args, _ = argsparser.parse_training_args()
    paths = get_paths(version, args, all_in_one_config=all_in_one_config)

    socket_url_client_array = set_up_drl_server(paths['all_in_one_parameters'])

    # check if simulations are booted
    wait_for_nodes(n_envs=args.n_envs, timeout=5)

    params = initialize_hyperparameters(paths, None, n_envs=args.n_envs, config_name='all_in_one_default')

    env = SubprocVecEnv(
        [make_all_in_one_envs(i, params=params, paths=paths, drl_server_url=socket_url_client_array,
                              num_envs=args.n_envs) for i in range(args.n_envs)], start_method='fork')

    eval_env = DummyVecEnv(
        [make_all_in_one_envs(0, params=params, paths=paths, train=False, drl_server_url=socket_url_client_array,
                              num_envs=args.n_envs, eval_episodes=eval_episodes)])


    load_path = os.path.join(paths['model'], 'vec_normalize.pkl')
    if os.path.isfile(load_path):
        env = VecNormalize.load(
            load_path=load_path, venv=env)
        eval_env = VecNormalize.load(
            load_path=load_path, venv=eval_env)
        print("Succesfully loaded VecNormalize object from pickle file..")
    else:
        env = VecNormalize(
            env, training=True,
            norm_obs=True, norm_reward=False, clip_reward=15)
        eval_env = VecNormalize(
            eval_env, training=True,
            norm_obs=True, norm_reward=False, clip_reward=15)

    eval_cb = EvalCallback(
        eval_env=eval_env, train_env=env,
        n_eval_episodes=eval_episodes, eval_freq=20000,
        log_path=paths['eval'], best_model_save_path=paths['model'], deterministic=True)

    # if args.agent == "AGENT_10": # TODO make it possible to set different architectures
    policy_kwargs = policy_kwargs_agent_13

    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs,
        gamma=params['gamma'], n_steps=params['n_steps'],
        ent_coef=params['ent_coef'], learning_rate=params['learning_rate'],
        vf_coef=params['vf_coef'], max_grad_norm=params['max_grad_norm'],
        gae_lambda=params['gae_lambda'], batch_size=params['m_batch_size'],
        n_epochs=params['n_epochs'], clip_range=params['clip_range'],
        tensorboard_log=paths.get('tb'), verbose=1
    )

    print("Start training...")

    if args.n is None:
        n_timesteps = 10000000
    else:
        n_timesteps = args.n

    # start training
    start = time.time()
    model.learn(total_timesteps=n_timesteps, callback=eval_cb)

    # training finished, close everything
    print(f'Time passed: {time.time() - start}s')
    print("training done!")
    env.close()
    eval_env.close()
    del model, env, eval_env
    sys.exit()
