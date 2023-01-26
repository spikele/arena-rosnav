#! /usr/bin/env python3

#NEW VERSION 2

from operator import is_
from random import randint
import gym
from gym import spaces
from gym.spaces import space
from typing import Union
from stable_baselines3.common.env_checker import check_env
import yaml
from rl_agent.utils.observation_collector import ObservationCollector

# from rl_agent.utils.CSVWriter import CSVWriter # TODO: @Elias: uncomment when csv-writer exists
from rl_agent.utils.reward import RewardCalculator
from rl_agent.utils.debug import timeit
from task_generator.tasks import ABSTask
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from flatland_msgs.srv import StepWorld, StepWorldRequest
from std_msgs.msg import Bool
import time
import math

from rl_agent.utils.debug import timeit
from task_generator.task_generator.tasks import *


class FlatlandGoalEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(
        self,
        ns: str,
        reward_fnc: str,
        is_action_space_discrete,
        safe_dist: float = None,
        goal_radius: float = 0.1,
        max_steps_per_episode=100,
        train_mode: bool = True,
        debug: bool = False,
        task_mode: str = "staged",
        PATHS: dict = dict(),
        extended_eval: bool = False,
        *args,
        **kwargs,
    ):
        """Default env
        Flatland yaml node check the entries in the yaml file, therefore other robot related parameters cound only be saved in an other file.
        TODO : write an uniform yaml paser node to handel with multiple yaml files.


        Args:
            task (ABSTask): [description]
            reward_fnc (str): [description]
            train_mode (bool): bool to differ between train and eval env during training
            is_action_space_discrete (bool): [description]
            safe_dist (float, optional): [description]. Defaults to None.
            goal_radius (float, optional): [description]. Defaults to 0.1.
            extended_eval (bool): more episode info provided, no reset when crashing
        """
        super(FlatlandGoalEnv, self).__init__()

        self.ns = ns
        try:
            # given every environment enough time to initialize, if we dont put sleep,
            # the training script may crash.
            ns_int = int(ns.split("_")[1])
            time.sleep(ns_int * 2)
        except Exception:
            rospy.logwarn(f"Can't not determinate the number of the environment, training script may crash!")

        # process specific namespace in ros system
        self.ns_prefix = "" if (ns == "" or ns is None) else "/" + ns + "/"

        if not debug:
            if train_mode:
                rospy.init_node(f"train_env_{self.ns}", disable_signals=False)
            else:
                rospy.init_node(f"eval_env_{self.ns}", disable_signals=False)

        self._extended_eval = extended_eval
        self._is_train_mode = rospy.get_param("/train_mode")
        self._is_action_space_discrete = is_action_space_discrete

        self.setup_by_configuration(PATHS["robot_setting"], PATHS["robot_as"])

        # observation collector
        self.observation_collector = ObservationCollector(
            self.ns,
            self._laser_num_beams,
            self._laser_max_range,
            external_time_sync=False,
        )


        achieved_goal_space = ObservationCollector._stack_spaces((
            spaces.Box(low=0, high=15, shape=(1,), dtype=np.float32),
            spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
        ))
        desired_goal_space = ObservationCollector._stack_spaces((
            spaces.Box(low=0, high=15, shape=(1,), dtype=np.float32),
            spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
        ))
        merged_observation_space = self.observation_collector.get_observation_space()
        """self.observation_space = spaces.Dict({
            "observation": merged_observation_space,
            "achieved_goal": achieved_goal_space,
            "desired_goal": desired_goal_space,
        })"""
        merged_observation_space = self.observation_collector.get_observation_space()
        self.observation_space = spaces.Dict(dict(
            observation=merged_observation_space,
            achieved_goal=achieved_goal_space,
            desired_goal=desired_goal_space,
        ))

        # csv writer # TODO: @Elias: uncomment when csv-writer exists
        # self.csv_writer=CSVWriter()
        # rospy.loginfo("======================================================")
        # rospy.loginfo("CSVWriter initialized.")
        # rospy.loginfo("======================================================")

        # reward calculator
        if safe_dist is None:
            safe_dist = self._robot_radius + 0.1

        self.reward_calculator = RewardCalculator(
            holonomic=self._holonomic,
            robot_radius=self._robot_radius,
            safe_dist=safe_dist,
            goal_radius=goal_radius,
            rule=reward_fnc,
            extended_eval=self._extended_eval,
        )

        # action agent publisher
        if self._is_train_mode:
            self.agent_action_pub = rospy.Publisher(f"{self.ns_prefix}cmd_vel", Twist, queue_size=1)
        else:
            self.agent_action_pub = rospy.Publisher(f"{self.ns_prefix}cmd_vel_pub", Twist, queue_size=1)

        # service clients
        if self._is_train_mode:
            self._service_name_step = f"{self.ns_prefix}step_world"
            self._sim_step_client = rospy.ServiceProxy(self._service_name_step, StepWorld)

        # instantiate task manager
        self.task = get_predefined_task(ns, mode=task_mode, start_stage=kwargs["curr_stage"], PATHS=PATHS)

        self._steps_curr_episode = 0
        self._episode = 0
        self._max_steps_per_episode = max_steps_per_episode
        self._last_action = np.array([0, 0, 0])  # linear x, linear y, angular z

        # for extended eval
        self._action_frequency = 1 / rospy.get_param("/robot_action_rate")
        self._last_robot_pose = None
        self._distance_travelled = 0
        self._safe_dist_counter = 0
        self._collisions = 0
        self._in_crash = False

        # NEW for goal env
        self.last_robot_position = None
        #self.goal_radius = goal_radius

        self._done_reasons = {
            "0": "Exc. Max Steps",
            "1": "Crash",
            "2": "Goal Reached",
        }
        self._done_hist = 3 * [0]

    def setup_by_configuration(self, robot_yaml_path: str, settings_yaml_path: str):
        """get the configuration from the yaml file, including robot radius, discrete action space and continuous action space.

        Args:
            robot_yaml_path (str): [description]
        """
        self._robot_radius = rospy.get_param("radius") + 0.25
        with open(robot_yaml_path, "r") as fd:
            robot_data = yaml.safe_load(fd)

            # get laser related information
            for plugin in robot_data["plugins"]:
                if plugin["type"] == "Laser" and plugin["name"] == "static_laser":
                    laser_angle_min = plugin["angle"]["min"]
                    laser_angle_max = plugin["angle"]["max"]
                    laser_angle_increment = plugin["angle"]["increment"]
                    self._laser_num_beams = int(round((laser_angle_max - laser_angle_min) / laser_angle_increment))
                    self._laser_max_range = plugin["range"]

        with open(settings_yaml_path, "r") as fd:
            setting_data = yaml.safe_load(fd)

            self._holonomic = setting_data["robot"]["holonomic"]

            if self._is_action_space_discrete:
                # self._discrete_actions is a list, each element is a dict with the keys ["name", 'linear','angular']
                assert not self._holonomic, "Discrete action space currently not supported for holonomic robots"
                self._discrete_acitons = setting_data["robot"]["discrete_actions"]
                self.action_space = spaces.Discrete(len(self._discrete_acitons))
            else:
                linear_range = setting_data["robot"]["continuous_actions"]["linear_range"]
                angular_range = setting_data["robot"]["continuous_actions"]["angular_range"]

                if not self._holonomic:
                    self.action_space = spaces.Box(
                        low=np.array([linear_range[0], angular_range[0]]),
                        high=np.array([linear_range[1], angular_range[1]]),
                        dtype=np.float32,   # CHANGED TO float32 from float
                    )
                else:
                    linear_range_x, linear_range_y = (
                        linear_range["x"],
                        linear_range["y"],
                    )
                    self.action_space = spaces.Box(
                        low=np.array(
                            [
                                linear_range_x[0],
                                linear_range_y[0],
                                angular_range[0],
                            ]
                        ),
                        high=np.array(
                            [
                                linear_range_x[1],
                                linear_range_y[1],
                                angular_range[1],
                            ]
                        ),
                        dtype=np.float32,   # CHANGED TO float32 from float
                    )

    def _pub_action(self, action: np.ndarray) -> Twist:
        assert len(action) == 3
        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.linear.y = action[1]
        action_msg.angular.z = action[2]
        self.agent_action_pub.publish(action_msg)

    def _translate_disc_action(self, action):
        assert not self._holonomic, "Discrete action space currently not supported for holonomic robots"
        new_action = np.array([])
        new_action = np.append(new_action, self._discrete_acitons[action]["linear"])
        new_action = np.append(new_action, self._discrete_acitons[action]["angular"])

        return new_action

    def _extend_action_array(self, action: np.ndarray) -> np.ndarray:
        if self._holonomic:
            assert (
                self._holonomic and len(action) == 3
            ), "Robot is holonomic but action with only two freedoms of movement provided"
            return action
        else:
            assert (
                not self._holonomic and len(action) == 2
            ), "Robot is non-holonomic but action with more than two freedoms of movement provided"
            return np.array([action[0], 0, action[1]])

    def step(self, action: np.ndarray):
        """
        done_reasons:   0   -   exceeded max steps
                        1   -   collision with obstacle
                        2   -   goal reached
        """
        if self._is_action_space_discrete:
            action = self._translate_disc_action(action)
        action = self._extend_action_array(action)

        self._pub_action(action)
        # print(f"Linear: {action[0]}, Angular: {action[1]}")
        self._steps_curr_episode += 1

        # wait for new observations
        merged_obs, obs_dict = self.observation_collector.get_observations(last_action=self._last_action)
        self._last_action = action

        # calculate reward
        reward, reward_info = self.reward_calculator.get_reward(
            obs_dict["laser_scan"],
            obs_dict["goal_in_robot_frame"],
            action=action,
            global_plan=obs_dict["global_plan"],
            robot_pose=obs_dict["robot_pose"],
        )
        # print(f"cum_reward: {reward}")
        done = reward_info["is_done"]

        # extended eval info
        if self._extended_eval:
            self._update_eval_statistics(obs_dict, reward_info)

        # update last robot position for goal env
        #self.last_robot_position = obs_dict["robot_pose"]

        # info
        info = {}

        # NEW
        info["obs_dict"] = obs_dict
        info["action"] = action

        if done:
            info["done_reason"] = reward_info["done_reason"]
            info["is_success"] = reward_info["is_success"]

        if self._steps_curr_episode > self._max_steps_per_episode:
            done = True
            info["done_reason"] = 0
            info["is_success"] = 0

        # for logging
        if self._extended_eval and done:
            info["collisions"] = self._collisions
            info["distance_travelled"] = round(self._distance_travelled, 2)
            info["time_safe_dist"] = self._safe_dist_counter * self._action_frequency
            info["time"] = self._steps_curr_episode * self._action_frequency

        if done:
            if sum(self._done_hist) == 10 and self.ns_prefix != "/eval_sim/":
                print(
                    f"[ns: {self.ns_prefix}] Last 10 Episodes: "
                    f"{self._done_hist[0]}x - {self._done_reasons[str(0)]}, "
                    f"{self._done_hist[1]}x - {self._done_reasons[str(1)]}, "
                    f"{self._done_hist[2]}x - {self._done_reasons[str(2)]}, "
                )
                self._done_hist = [0] * 3
            self._done_hist[int(info["done_reason"])] += 1

        #return merged_obs, reward, done, info
        merged_goal_obs = {}
        merged_goal_obs["observation"] = merged_obs
        #if self.last_robot_position != None:
        #    achieved_rho, achieved_theta = ObservationCollector._get_goal_pose_in_robot_frame(obs_dict["robot_pose"], self.last_robot_position)
        #else:
        #    achieved_rho, achieved_theta = [0,0]
        achieved_rho, achieved_theta = [0,0]
        merged_goal_obs["achieved_goal"] = np.array([achieved_rho, achieved_theta])
        merged_goal_obs["desired_goal"] = np.array(obs_dict["goal_in_robot_frame"])

        self.last_robot_position = obs_dict["robot_pose"]

        #reward_binary = self.compute_reward(merged_goal_obs["achieved_goal"], merged_goal_obs["desired_goal"], {})

        return merged_goal_obs, reward, done, info
        #return merged_goal_obs, reward_binary, done, info

    def reset(self):
        # set task
        # regenerate start position end goal position of the robot and change the obstacles accordingly
        self._episode += 1
        self.agent_action_pub.publish(Twist())
        if self._is_train_mode:
            self._sim_step_client()
        self.task.reset()
        self.reward_calculator.reset()
        self._steps_curr_episode = 0
        self._last_action = np.array([0, 0, 0])

        # extended eval info
        if self._extended_eval:
            self._last_robot_pose = None
            self._distance_travelled = 0
            self._safe_dist_counter = 0
            self._collisions = 0

        self.last_robot_position = None

        #obs, _ = self.observation_collector.get_observations()
        #return obs  # reward, done, info can't be included
        merged_obs, obs_dict = self.observation_collector.get_observations()
        merged_goal_obs = {}
        merged_goal_obs["observation"] = merged_obs
        merged_goal_obs["achieved_goal"] = np.array([0, 0])
        merged_goal_obs["desired_goal"] = np.array(obs_dict["goal_in_robot_frame"])
        return merged_goal_obs



    '''def compute_reward(self, achieved_goal, desired_goal, info):
        """Binary reward: 0 if reached goal, else -1. Does not consider crashes."""
        #x_ach = achieved_goal[0] * np.cos(achieved_goal[1])
        #y_ach = achieved_goal[0] * np.sin(achieved_goal[1])
        #x_des = desired_goal[0] * np.cos(desired_goal[1])
        #y_des = desired_goal[0] * np.sin(desired_goal[1])
        
        """Distance between polar coordinates"""
        #distance = (achieved_goal[0] ** 2 + desired_goal[0] ** 2 - 2*achieved_goal[0]*desired_goal[0]*np.cos(achieved_goal[0]-desired_goal[1])) ** 0.5

        #print(achieved_goal.shape)
        print(achieved_goal[0])
        print(achieved_goal[1])
        if len(achieved_goal) > 2:
            print(achieved_goal[:,0])
            print(achieved_goal[:,1])

        return 0 if (achieved_goal[0] ** 2 + desired_goal[0] ** 2 - 2*achieved_goal[0]*desired_goal[0]*np.cos(achieved_goal[0]-desired_goal[1])) ** 0.5 < self.goal_radius else -1
'''
    # NEW
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on an a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in info and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution (the robot position after a step relative to before)
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve (the subgoal position relative to the robot position)
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """
        #print("achieved_goal: " + str(achieved_goal))
        #print("desired_goal: " + str(desired_goal))
        #print("achieved_goal length: " + str(len(achieved_goal)))
        #print("desired_goal length: " + str(len(desired_goal)))
        #print("info length: " + str(len(info)))
        #for key in info:
        #    print("info key: " + str(key))
        #print("info keys: ")
        #print(info[0].keys())
        #print("info[0]['obs_dict']: " + str(info[0]["obs_dict"]))

        length = len(desired_goal)

        reward = np.zeros(length)

        for i in range(length):
            reward[i], _ = self.reward_calculator.get_reward(
                info[i]["obs_dict"]["laser_scan"],
                info[i]["obs_dict"]["goal_in_robot_frame"], # this is the same as desired_goal
                action=info[i]["action"],
                global_plan=info[i]["obs_dict"]["global_plan"],
                robot_pose=info[i]["obs_dict"]["robot_pose"], # this has a connection to achieved_goal: achieved goal is robot_pose relative to the last robot_pose
            )
            
        """get_reward_args = {info[:]["obs_dict"]["laser_scan"],
                    info[:]["obs_dict"]["goal_in_robot_frame"],
                    info[:]["action"],
                    info[:]["obs_dict"]["global_plan"],
                    info[:]["obs_dict"]["robot_pose"]}"""
        """laser_scan = info[:]["obs_dict"]["laser_scan"]
        goal_in_robot_frame = info[:]["obs_dict"]["goal_in_robot_frame"]
        action = info[:]["action"]
        global_plan = info[:]["obs_dict"]["global_plan"]
        robot_pose = info[:]["obs_dict"]["robot_pose"]"""



        #reward = np.array(list(map(self.reward_calculator.get_reward[0], get_reward_args)))

        """reward, reward_info = self.reward_calculator.get_reward(
                laser_scan,
                goal_in_robot_frame, # this is the same as desired_goal
                action=action,
                global_plan=global_plan,
                robot_pose=robot_pose, # this has a connection to achieved_goal: achieved goal is robot_pose relative to the last robot_pose
            )"""

        #reward, reward_info = self.reward_calculator.get_reward(
        #    info["obs_dict"]["laser_scan"],
        #    info["obs_dict"]["goal_in_robot_frame"], # this is the same as desired_goal
        #    action=info["action"],
        #    global_plan=info["obs_dict"]["global_plan"],
        #    robot_pose=info["obs_dict"]["robot_pose"], # this has a connection to achieved_goal: achieved goal is robot_pose relative to the last robot_pose
        #)
        return reward


    def close(self):
        pass

    def _update_eval_statistics(self, obs_dict: dict, reward_info: dict):
        """
        Updates the metrics for extended eval mode

        param obs_dict (dict): observation dictionary from ObservationCollector.get_observations(),
            necessary entries: 'robot_pose'
        param reward_info (dict): dictionary containing information returned from RewardCalculator.get_reward(),
            necessary entries: 'crash', 'safe_dist'
        """
        # distance travelled
        if self._last_robot_pose is not None:
            self._distance_travelled += FlatlandGoalEnv.get_distance(self._last_robot_pose, obs_dict["robot_pose"])

        # collision detector
        if "crash" in reward_info:
            if reward_info["crash"] and not self._in_crash:
                self._collisions += 1
                # when crash occures, robot strikes obst for a few consecutive timesteps
                # we want to count it as only one collision
                self._in_crash = True
        else:
            self._in_crash = False

        # safe dist detector
        if "safe_dist" in reward_info and reward_info["safe_dist"]:
            self._safe_dist_counter += 1

        self._last_robot_pose = obs_dict["robot_pose"]

    @staticmethod
    def get_distance(pose_1: Pose2D, pose_2: Pose2D):
        return math.hypot(pose_2.x - pose_1.x, pose_2.y - pose_1.y)


if __name__ == "__main__":

    rospy.init_node("flatland_gym_env", anonymous=True, disable_signals=False)
    print("start")

    flatland_env = FlatlandGoalEnv()
    rospy.loginfo("======================================================")
    rospy.loginfo("CSVWriter initialized.")
    rospy.loginfo("======================================================")
    check_env(flatland_env, warn=True)

    # init env
    obs = flatland_env.reset()

    # run model
    n_steps = 200
    for _ in range(n_steps):
        # action, _states = model.predict(obs)
        action = flatland_env.action_space.sample()

        obs, rewards, done, info = flatland_env.step(action)

        time.sleep(0.1)
