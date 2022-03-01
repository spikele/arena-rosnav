#! /usr/bin/env python3
import numpy as np
import gym
from gym import spaces
from gym.spaces import space
from stable_baselines3.common.env_checker import check_env
import rospy
import time
import math


from geometry_msgs.msg import Pose2D, PoseStamped
#from flatland_msgs.srv import StepWorld, StepWorldRequest
from std_srvs.srv import Empty, EmptyRequest

from stable_baselines3.common.env_checker import check_env

from arena_navigation.arena_waypoint_generator.scripts.drl.rl_agent.envs.observation import *
from arena_navigation.arena_waypoint_generator.scripts.drl.rl_agent.envs.reward import *

from arena_navigation.arena_waypoint_generator.scripts.drl.task_generator.tasks import *

class Subgoal_env(gym.Env):
    def __init__(
        self,
        ns: str,
        reward_fnc: str,
        safe_dist: float = None,
        extended_eval: bool = False,
        task_mode: str = "staged",
        max_steps_per_episode=70,
        PATHS: dict = dict(),
        goal_radius: float = 0.7,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.degree = 180
        self.n = int(self.degree/10)
        self.radiant = (self.degree/180)*np.pi

        self.ns = ns
        self.ns_prefix = "" if (ns == "" or ns is None) else "/" + ns + "/"
        self._extended_eval = extended_eval

        #self.action_space = spaces.Discrete(self.n+1)
        self.action_space = spaces.MultiDiscrete([self.n,2]) #the first action for angle to get a subgoal, the second one for mode of subgoals
        self.obs_observation = Observation(ns=ns, PATHS=PATHS)
        self.observation_space = (self.obs_observation.get_observation_space())
        
        self.planing_horizon = self.obs_observation.get_lidar_range()
        self.obs_reward = Reward(safe_dist=safe_dist, goal_radius=goal_radius, extended_eval=self._extended_eval, planing_horizon=self.planing_horizon, n=self.n)

        self.agent_action_pub = rospy.Publisher(f"{self.ns_prefix}subgoal", PoseStamped, queue_size=1)
        #self._last_action = None
        self.action_in_radius = None

        self._steps_curr_episode = 0
        self._episode = 0
        self._max_steps_per_episode_unit = max_steps_per_episode
        
        #self.task = get_predefined_task(ns, mode=task_mode, start_stage=kwargs["curr_stage"], PATHS=PATHS)
        
        #scenarios_json_path = '/home/baoduc/arena_ws/src/arena-rosnav/simulator_setup/scenarios/empty_map.json'
       
        #paths = {"scenario": scenarios_json_path}
  
        self.task = get_predefined_task(ns, mode=task_mode, start_stage=kwargs["curr_stage"], PATHS=PATHS)

        self._action_frequency = 1 / rospy.get_param("/robot_action_rate")
        self._last_robot_pose = None
        self._distance_travelled = 0
        self._safe_dist_counter = 0
        self._collisions = 0
        self._in_crash = False

        self._done_reasons = {
            "0": "Exc. Max Steps",
            "1": "Crash",
            "2": "Goal Reached",
        }
        self._done_hist = 3 * [0]

        self._subgoal = Pose2D()
        self.subgoal_tolerance = goal_radius/2
        self.obs_dict = dict()
        self._episode_ref = 0

        self.clear_costmaps_srv = rospy.ServiceProxy(
            '/move_base/clear_costmaps', Empty)

        #self._service_name_step = f"{self.ns_prefix}step_world"
        #self._sim_step_client = rospy.ServiceProxy(
            #self._service_name_step, StepWorld
        #)

    def step(self, actions):
        self._pub_action(actions)

        obs, obs_dict = self.obs_observation.get_observations()#last_action=self._last_action)
        self.obs_dict = obs_dict
        #self._last_action = actions

        if self._steps_curr_episode == 0:
            global_plan_length = self.obs_observation.get_global_plan_length()
            if global_plan_length != 0:
                self._max_steps_per_episode = int(self._max_steps_per_episode_unit*global_plan_length/self.planing_horizon)
            #print(f"length = {global_plan_length:.2f}")
            #print(f"steps = {self._max_steps_per_episode}")

        self._steps_curr_episode += 1

        reward, reward_info = self.obs_reward.get_reward(
                obs_dict["laser_scan"], 
                obs_dict["goal_in_robot_frame"], 
                actions=actions, 
                global_plan=obs_dict["global_plan"], 
                robot_pose=obs_dict["robot_pose"],
                scan_angle=obs_dict["scan_angle"],
                action_in_radius=self.action_in_radius,
                subgoal=self._subgoal,
                goal = self.obs_observation.get_goal_pose(),
            )
        done = reward_info["is_done"]

        #if self._steps_curr_episode == 100:
            #print(f"step = {self._steps_curr_episode}")
            #print(f"reward = {reward}")
            #print(self.obs_observation.get_global_plan())

        if self._extended_eval:
            self._update_eval_statistics(obs_dict, reward_info)

        info = {}

        if done:
            info["done_reason"] = reward_info["done_reason"]
            info["is_success"] = reward_info["is_success"]

        if self._steps_curr_episode > self._max_steps_per_episode:
            done = True
            info["done_reason"] = 0
            info["is_success"] = 0

        if self._extended_eval and done:
            info["collisions"] = self._collisions
            info["distance_travelled"] = round(self._distance_travelled, 2)
            info["time_safe_dist"] = (self._safe_dist_counter * self._action_frequency)
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
        return obs, reward, done, info

    def reset(self):
        self._episode += 1
        action_msg = PoseStamped()
        action_msg.header.stamp = rospy.Time.now()
        action_msg.header.frame_id = "map"
        action_msg.pose.position.x = self.obs_observation.get_goal_pose().x
        action_msg.pose.position.y = self.obs_observation.get_goal_pose().y
        self.agent_action_pub.publish(action_msg)
        #self._sim_step_client()
        self.task.reset()
        self.obs_reward.reset_reward_()
        #print(self._steps_curr_episode)
        self._steps_curr_episode = 0
        #self._last_action = None

        if self._extended_eval:
            self._last_robot_pose = None
            self._distance_travelled = 0
            self._safe_dist_counter = 0
            self._collisions = 0

        obs, _ = self.obs_observation.get_observations()

        self.clear_costmaps_srv(EmptyRequest())
        return obs

    def close(self):
        pass

    def _update_eval_statistics(self, obs_dict: dict, reward_info: dict):
        if self._last_robot_pose is not None:
            self._distance_travelled += Subgoal_env.get_distance(self._last_robot_pose, obs_dict["robot_pose"])

        if "crash" in reward_info:
            if reward_info["crash"] and not self._in_crash:
                self._collisions += 1
                self._in_crash = True
        else:
            self._in_crash = False

        if "safe_dist" in reward_info and reward_info["safe_dist"]:
            self._safe_dist_counter += 1

        self._last_robot_pose = obs_dict["robot_pose"]

    def _pub_action(self, actions: int) -> Pose2D:
        action_msg = PoseStamped()
        action_msg.header.stamp = rospy.Time.now()
        action_msg.header.frame_id = "map"
        _robot_pose = self.obs_observation.get_robot_pose()
        _robot_orientierung = _robot_pose.theta
        _goal_pose = self.obs_observation.get_goal_pose()

        action_in_radius = -self.radiant/2 + (self.radiant/self.n)*actions[0]
        self.action_in_radius = action_in_radius

        if self.get_distance(_goal_pose, _robot_pose) <= self.planing_horizon:
            action_msg.pose.position.x = self.obs_observation.get_goal_pose().x
            action_msg.pose.position.y = self.obs_observation.get_goal_pose().y
            self.agent_action_pub.publish(action_msg)
            return

        #if actions[1] == 0:
         #   action_msg.pose.position.x = _robot_pose.x
         #   action_msg.pose.position.y = _robot_pose.y

         #   self._subgoal.x = action_msg.pose.position.x
         #   self._subgoal.y = action_msg.pose.position.y
         #   self.agent_action_pub.publish(action_msg)
         #   return

        if actions[1] == 0: #new subgoal won't be published 
            if self.get_distance(self._subgoal, _robot_pose) > self.subgoal_tolerance:
                if self.get_distance(self._subgoal, _robot_pose) <= self.planing_horizon:
                    if self._episode == self._episode_ref:
                        action_msg.pose.position.x = self._subgoal.x
                        action_msg.pose.position.y = self._subgoal.y
                        self.agent_action_pub.publish(action_msg)
                        return
                    else:
                        self._episode_ref = self._episode
                        action_msg.pose.position.x = _robot_pose.x
                        action_msg.pose.position.y = _robot_pose.y

                        self._subgoal.x = action_msg.pose.position.x
                        self._subgoal.y = action_msg.pose.position.y
                        self.agent_action_pub.publish(action_msg)
                        return

        try:
            global_plan = self.obs_dict["global_plan"]
            subgoal_id = 0
            for i in range(len(global_plan)):
                wp = Pose2D()
                wp.x = global_plan[i][0]
                wp.y = global_plan[i][1]
                dist_to_robot = self.get_distance(wp, _robot_pose)
                if (dist_to_robot<self.planing_horizon+self.subgoal_tolerance) and (dist_to_robot>self.planing_horizon-self.subgoal_tolerance):
                    if i > subgoal_id:
                        subgoal_id = i

            subgoal = global_plan[subgoal_id]
            y_relative = subgoal[1] - _robot_pose.y
            x_relative = subgoal[0] - _robot_pose.x
            phi = (np.arctan2(y_relative, x_relative) - _robot_pose.theta + 4 * np.pi) % (2 * np.pi) - np.pi   

            action_msg.pose.position.x = _robot_pose.x + self.planing_horizon*np.cos(_robot_orientierung+action_in_radius+phi-np.pi)
            action_msg.pose.position.y = _robot_pose.y + self.planing_horizon*np.sin(_robot_orientierung+action_in_radius+phi-np.pi)

            self._subgoal.x = action_msg.pose.position.x
            self._subgoal.y = action_msg.pose.position.y

            self.agent_action_pub.publish(action_msg)
        except:
            action_msg.pose.position.x = _robot_pose.x + self.planing_horizon*np.cos(_robot_orientierung+action_in_radius)
            action_msg.pose.position.y = _robot_pose.y + self.planing_horizon*np.sin(_robot_orientierung+action_in_radius)

            self._subgoal.x = action_msg.pose.position.x
            self._subgoal.y = action_msg.pose.position.y

            self.agent_action_pub.publish(action_msg)

    @staticmethod
    def get_distance(pose_1: Pose2D, pose_2: Pose2D):
        return math.hypot(pose_2.x - pose_1.x, pose_2.y - pose_1.y)
    

if __name__ == "__main__":

    rospy.init_node("subgoal_gym_env", anonymous=True, disable_signals=False)
    print("start")

    subgoal_env = Subgoal_env()
    rospy.loginfo("======================================================")
    rospy.loginfo("CSVWriter initialized.")
    rospy.loginfo("======================================================")
    check_env(subgoal_env, warn=True)

    # init env
    obs = subgoal_env.reset()

    # run model
    n_steps = 100
    for _ in range(n_steps):
        action = subgoal_env.action_space.sample()

        #obs, rewards, done, info = subgoal_env.step(action)

        print(action)

        time.sleep(0.1)