from typing import Tuple

import gym, rospy
import os
import rospkg
import torch as th
import yaml

from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from geometry_msgs.msg import Pose2D
from rl_agent.utils.observation_collector import ObservationCollector

""" 
_RS: Robot state size - placeholder for robot related inputs to the NN
_L: Number of laser beams - placeholder for the laser beam data 
"""
action_in_obs = rospy.get_param(
    "actions_in_obs",
    default=False
)
print("ACTIONS IN OBSERVATION SPACE IN FEATURE EXTRACTORS:" + str(action_in_obs))
#if not rospy.get_param("actions_in_obs", default=False):
if not action_in_obs:
    _RS = 2  # robot state size
else:
    _RS = 2 + 3  # rho, theta, linear x, linear y, angular z
robot_model = rospy.get_param("model")

ROBOT_SETTING_PATH = rospkg.RosPack().get_path("simulator_setup")
yaml_ROBOT_SETTING_PATH = os.path.join(
    ROBOT_SETTING_PATH, "robot", f"{robot_model}.model.yaml"
)

with open(yaml_ROBOT_SETTING_PATH, "r") as fd:
    robot_data = yaml.safe_load(fd)
    for plugin in robot_data["plugins"]:
        if plugin["type"] == "Laser":
            laser_angle_min = plugin["angle"]["min"]
            laser_angle_max = plugin["angle"]["max"]
            laser_angle_increment = plugin["angle"]["increment"]
            _L = int(
                round(
                    (laser_angle_max - laser_angle_min) / laser_angle_increment
                )
            )  # num of laser beams
            break


class MLP_ARENA2D(nn.Module):
    """
    Custom Multilayer Perceptron for policy and value function.
    Architecture was taken as reference from: https://github.com/ignc-research/arena2D/tree/master/arena2d-agents.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 32,
        last_layer_dim_vf: int = 32,
    ):
        super(MLP_ARENA2D, self).__init__()

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Body network
        self.body_net = nn.Sequential(
            nn.Linear(_L + _RS, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        body_x = self.body_net(features)
        return self.policy_net(body_x), self.value_net(body_x)


class EXTRACTOR_1(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.
    Architecture was taken as reference from: https://arxiv.org/abs/1808.03841

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 128
    ):
        super(EXTRACTOR_1, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        laser_scan = th.unsqueeze(observations[:, :-_RS], 1)
        robot_state = observations[:, -_RS:]

        extracted_features = self.fc_1(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_2(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.
    Architecture was taken as reference from: https://arxiv.org/abs/1808.03841
    (DRL_LOCAL_PLANNER)

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 128
    ):
        super(EXTRACTOR_2, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        laser_scan = th.unsqueeze(observations[:, :-_RS], 1)
        robot_state = observations[:, -_RS:]

        extracted_features = self.fc_1(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_3(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network to serve as feature extractor ahead of the policy and value network.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 128
    ):
        super(EXTRACTOR_3, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # tensor_forward = th.as_tensor(observation_space.sample()[None]).float()
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc_1 = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
        )

        self.fc_2 = nn.Sequential(nn.Linear(256, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor),
            extracted features by the network
        """
        laser_scan = th.unsqueeze(observations[:, :-_RS], 1)
        robot_state = observations[:, -_RS:]

        extracted_features = self.fc_2(self.fc_1(self.cnn(laser_scan)))
        # return self.fc_2(features)
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_4(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.
    Architecture was taken as reference from: https://github.com/ethz-asl/navrep
    (CNN_NAVREP)

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 32
    ):
        super(EXTRACTOR_4, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 9, 4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 6, 4),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, :-_RS], 1)
        robot_state = observations[:, -_RS:]

        extracted_features = self.fc(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_5(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 32
    ):
        super(EXTRACTOR_5, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, :-_RS], 1)
        robot_state = observations[:, -_RS:]

        extracted_features = self.fc(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_6(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 32
    ):
        super(EXTRACTOR_6, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations[:, :-_RS], 1)
        robot_state = observations[:, -_RS:]

        extracted_features = self.fc(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_6_HER(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head. Version for HER.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Dict, features_dim: int = 32
    ):
        super(EXTRACTOR_6_HER, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, _L)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        laser_scan = th.unsqueeze(observations["observation"][:, :-_RS], 1)

        
        #robot_state = obs[:, -_RS:]
        """desired_pose = Pose2D()
        desired_pose.x = observations["desired_goal"][0]
        desired_pose.y = observations["desired_goal"][1]
        desired_pose.theta = 0
        achieved_pose = Pose2D()
        achieved_pose.x = observations["achieved_goal"][0]
        achieved_pose.y = observations["achieved_goal"][1]
        achieved_pose.theta = 0

        goal_pose_i_r_f = _get_goal_pose_in_robot_frame(desired_pose, achieved_pose)

        if action_in_obs:
            robot_state = th.cat(goal_pose_i_r_f, observations["observation"][:, -3:])
        else:
            robot_state = th.tensor(goal_pose_i_r_f)"""

        length = observations["observation"].shape[0]
        #length = 1

        goal_poses = []

        #print(observations.keys())
        #print(observations["observation"].shape)
        #print(observations["observation"][:].shape)
        #print(observations["observation"][:, :-_RS].shape)
        #print(observations["observation"][:-_RS].shape)
        #print(observations["desired_goal"].shape)

        for i in range(length):
            goal_pos = Pose2D()
            goal_pos.x = observations["desired_goal"][i, 0]
            goal_pos.y = observations["desired_goal"][i, 1]
            goal_pos.theta = observations["desired_goal"][i, 2]
            robot_pos = Pose2D()
            robot_pos.x = observations["achieved_goal"][i, 0]
            robot_pos.y = observations["achieved_goal"][i, 1]
            robot_pos.theta = observations["achieved_goal"][i, 2]


            y_relative = goal_pos.y - robot_pos.y
            x_relative = goal_pos.x - robot_pos.x
            rho = (x_relative ** 2 + y_relative ** 2) ** 0.5
            theta = (
                th.arctan2(y_relative, x_relative) - robot_pos.theta + 4 * th.pi
            ) % (2 * th.pi) - th.pi

            goal_poses.append([rho, theta])

        goal_poses_tensor = th.tensor(goal_poses, device=th.device('cuda'))
        #print(goal_poses_tensor.shape)
        #print(observations["observation"][:, -3:].shape)

        if action_in_obs:
            robot_state = th.cat((goal_poses_tensor, observations["observation"][:, -3:]), 1)
        else:
            robot_state = goal_poses_tensor

        #print(robot_state.shape)
        #robot_state = observations["observation"][:, -_RS:]

        extracted_features = self.fc(self.cnn(laser_scan))
        return th.cat((extracted_features, robot_state), 1)


class EXTRACTOR_6_FRAME_STACK(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head. Version for frame stacking.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Box, features_dim: int = 32
    ):
        super(EXTRACTOR_6_FRAME_STACK, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, _L*4)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        one_obs = _L + _RS
        #print(_RS)
        #laser_scan = th.unsqueeze(observations[:, :one_obs-_RS], 1)
        #print("laser_scan shape:" + str(laser_scan.shape))
        
        laser_scans = th.unsqueeze(th.cat((observations[:, 0:_L], observations[:, one_obs:one_obs+_L], observations[:, 2*one_obs:2*one_obs+_L], observations[:, 3*one_obs:3*one_obs+_L]), 1), 1)
        #print("laser_scans shape:" + str(laser_scans.shape))
        robot_state = observations[:, -_RS:]

        #extracted_features = self.fc(self.cnn(laser_scan))
        extracted_features = self.fc(self.cnn(laser_scans))
        return th.cat((extracted_features, robot_state), 1)

class EXTRACTOR_6_FRAME_STACK_HER(BaseFeaturesExtractor):
    """
    Custom Convolutional Neural Network (Nature CNN) to serve as feature extractor ahead of the policy and value head.  Version for HER combined with frame stacking.

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self, observation_space: gym.spaces.Dict, features_dim: int = 32
    ):
        super(EXTRACTOR_6_FRAME_STACK_HER, self).__init__(observation_space, features_dim + _RS)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 8, 4),
            nn.ReLU(),
            nn.Conv1d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            tensor_forward = th.randn(1, 1, _L*4)
            n_flatten = self.cnn(tensor_forward).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        :return: (th.Tensor) features,
            extracted features by the network
        """

        one_obs = _L + _RS
        #print(_RS)
        #laser_scan = th.unsqueeze(observations[:, :one_obs-_RS], 1)
        #print("laser_scan shape:" + str(laser_scan.shape))
        obs = observations["observation"]
        
        laser_scans = th.unsqueeze(th.cat((obs[:, 0:_L], obs[:, one_obs:one_obs+_L], obs[:, 2*one_obs:2*one_obs+_L], obs[:, 3*one_obs:3*one_obs+_L]), 1), 1)
        #print("laser_scans shape:" + str(laser_scans.shape))

        #robot_state = obs[:, -_RS:]
        #length = len(observations)
        length = obs.shape[0]

        #print(observations.keys())
        #print(observations["observation"].shape)
        #print(observations["desired_goal"].shape)

        goal_poses = []

        for i in range(length):
            goal_pos = Pose2D()
            goal_pos.x = observations["desired_goal"][i, -3]
            goal_pos.y = observations["desired_goal"][i, -2]
            goal_pos.theta = observations["desired_goal"][i, -1]
            robot_pos = Pose2D()
            robot_pos.x = observations["achieved_goal"][i, -3]
            robot_pos.y = observations["achieved_goal"][i, -2]
            robot_pos.theta = observations["achieved_goal"][i, -1]


            y_relative = goal_pos.y - robot_pos.y
            x_relative = goal_pos.x - robot_pos.x
            rho = (x_relative ** 2 + y_relative ** 2) ** 0.5
            theta = (
                th.arctan2(y_relative, x_relative) - robot_pos.theta + 4 * th.pi
            ) % (2 * th.pi) - th.pi

            goal_poses.append([rho, theta])

        goal_poses_tensor = th.tensor(goal_poses, device=th.device('cuda'))

        if action_in_obs:
            robot_state = th.cat((goal_poses_tensor, observations["observation"][:, -3:]), 1)
        else:
            robot_state = goal_poses_tensor

        #extracted_features = self.fc(self.cnn(laser_scan))
        extracted_features = self.fc(self.cnn(laser_scans))
        return th.cat((extracted_features, robot_state), 1)
