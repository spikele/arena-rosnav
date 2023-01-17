#!/usr/bin/env python3
import os
import pickle
import rospy
import rospkg
import sys
from datetime import datetime as dt

from stable_baselines3 import PPO

from flatland_msgs.srv import StepWorld, StepWorldRequest
from rospy.exceptions import ROSException
from std_msgs.msg import Int16, Bool

from rl_agent.base_agent_wrapper import BaseDRLAgent

robot_model = rospy.get_param("model")
""" TEMPORARY GLOBAL CONSTANTS """
NS_PREFIX = ""
TRAINED_MODELS_DIR = os.path.join(
    rospkg.RosPack().get_path("arena_local_planner_drl"), "agents_thesis"
)
DEFAULT_ACTION_SPACE = os.path.join(
    rospkg.RosPack().get_path("arena_local_planner_drl"),
    "configs",
    f"default_settings_{robot_model}.yaml",
)
RECORDINGS_DIR = os.path.join(
    rospkg.RosPack().get_path("arena_local_planner_drl"), "recordings"
)


class DeploymentDRLAgent(BaseDRLAgent):
    def __init__(
        self,
        agent_name: str,
        ns: str = None,
        robot_name: str = None,
        action_space_path: str = DEFAULT_ACTION_SPACE,
        *args,
        **kwargs,
    ) -> None:
        """Initialization procedure for the DRL agent node.

        Args:
            agent_name (str):
                Agent name (directory has to be of the same name)
            robot_name (str, optional):
                Robot specific ROS namespace extension. Defaults to None.
            ns (str, optional):
                Simulation specific ROS namespace. Defaults to None.
            action_space_path (str, optional):
                Path to yaml file containing action space settings.
                Defaults to DEFAULT_ACTION_SPACE.
        """
        print("start of init")

        self.epoch_obs = []
        self.epoch_act = []
        self.episode_obs = []
        self.episode_act = []

        self.episode_info = []

        self.total_count = 0
        self.success_count = 0
        self.crash_count = 0

        self._is_train_mode = rospy.get_param("/train_mode")
        if not self._is_train_mode:
            rospy.init_node(f"DRL_local_planner", anonymous=True)

        self.name = agent_name

        hyperparameter_path = os.path.join(
            TRAINED_MODELS_DIR, self.name, "hyperparameters.json"
        )
        super().__init__(
            ns,
            robot_name,
            hyperparameter_path,
            action_space_path,
        )
        self.setup_agent()

        if self._is_train_mode:
            # step world to fast forward simulation time
            self._service_name_step = f"{self._ns}step_world"
            self._sim_step_client = rospy.ServiceProxy(
                self._service_name_step, StepWorld
            )

        self._reset_sub = rospy.Subscriber("/scenario_reset",
            Int16,
            self.end_episode,
            tcp_nodelay=True,)

        os.makedirs(RECORDINGS_DIR, exist_ok=True)

        WORLD_PATH_PARAM = rospy.get_param("world_path")
        MAP = os.path.split(os.path.split(WORLD_PATH_PARAM)[0])[1]
        START_TIME = dt.now().strftime("%Y_%m_%d__%H_%M")

        self.RECORDING_NAME = "recording_" + "_" + MAP + "_" + "ROSNAV" + START_TIME + ".dictionary"

    def setup_agent(self) -> None:
        """Loads the trained policy and when required the VecNormalize object."""
        model_file = os.path.join(
            TRAINED_MODELS_DIR, self.name, "best_model.zip"
        )
        vecnorm_file = os.path.join(
            TRAINED_MODELS_DIR, self.name, "vec_normalize.pkl"
        )

        assert os.path.isfile(
            model_file
        ), f"Compressed model cannot be found at {model_file}!"

        if self._agent_params["normalize"]:
            assert os.path.isfile(
                vecnorm_file
            ), f"VecNormalize file cannot be found at {vecnorm_file}!"

            with open(vecnorm_file, "rb") as file_handler:
                vec_normalize = pickle.load(file_handler)
            self._obs_norm_func = vec_normalize.normalize_obs

        self._agent = PPO.load(model_file).policy

    def run(self) -> None:
        """Loop for running the agent until ROS is shutdown.
        
        Note:
            Calls the 'step_world'-service for fast-forwarding the \
            simulation time in training mode. The simulation is forwarded \
            by action_frequency seconds. Otherwise, communicates with \
            the ActionPublisher node in order to comply with the specified \
            action publishing rate.
        """
        while not rospy.is_shutdown():
            # if self._is_train_mode:
            #     self.call_service_takeSimStep(self._action_frequency)
            # else:
            #     self._wait_for_next_action_cycle()
            obs = self.get_observations()[0]
            action = self.get_action(obs)

            self.episode_obs.append(obs)
            self.episode_act.append(action)

            self.publish_action(action)

        # when shutdown, save epoch again with the latest episode
        self.end_episode(-1)
        print((str(self.total_count)) + "episodes")
        """print(str(self.success_count) + "successes (" + str((self.success_count/self.total_count)*100) + "%)")
        print(str(self.crash_count) + "crashes (" + str((self.crash_count/self.total_count)*100) + "%)")"""

    def end_episode(self, msg_nr) -> None:
        self.total_count += 1
        """was_crash = False
        
        for i in range(0, len(self.episode_info)):
            info = self.episode_info[i]

            
                # TODO: IDEA: instead of cutting off when done, omit episode when there is a crash because it is probably counterproductive to learn from it
            if info["is_done"]:
                self.episode_act = self.episode_act[0:i+1]
                self.episode_info = self.episode_info[0:i+1]
                self.episode_obs = self.episode_obs[0:i+2]

                if info["done_reason"] == 2:
                    print("cut episode because it was a success")
                    self.success_count += 1
                elif info["done_reason"] == 1:
                    print("cut episode because there was a crash")
                    self.crash_count += 1
                    was_crash = True
                else:
                    print("ELSE (should not occur)")
                break"""

        if msg_nr != -1:
            num = msg_nr.data
        else:
            num = msg_nr
        print("end of the " + str(num) + ". episode, size: " + str(len(self.episode_obs)))
        
        self.epoch_obs.append(self.episode_obs)
        self.epoch_act.append(self.episode_act[:-1])
        """if not was_crash:
            self.epoch_obs.append(self.episode_obs)
            self.epoch_act.append(self.episode_act)"""

        self.episode_obs = []
        self.episode_act = []
        self.episode_info = []
        
        #with open('/home/liam/BC_recordings/recording.dictionary', 'wb') as file:

        with open(os.path.join(RECORDINGS_DIR, self.RECORDING_NAME), 'wb') as file:
            pickle.dump([self.epoch_obs, self.epoch_act], file)
            file.close()
            #print("saved observations")

    def _wait_for_next_action_cycle(self) -> None:
        """Stops the loop until a trigger message is sent by the ActionPublisher

        Note:
            Only use this method in combination with the ActionPublisher node!
            That node is only booted when training mode is off.
        """
        try:
            rospy.wait_for_message(f"{self._ns_robot}next_cycle", Bool)
        except ROSException:
            pass

    def call_service_takeSimStep(self, t: float = None) -> None:
        """Fast-forwards the simulation time.

        Args:
            t (float, optional):
                Time in seconds. When t is None, time is forwarded by 'step_size' s.
                Defaults to None.
        """
        request = StepWorldRequest() if t is None else StepWorldRequest(t)

        try:
            response = self._sim_step_client(request)
            rospy.logdebug("step service=", response)
        except rospy.ServiceException as e:
            rospy.logdebug("step Service call failed: %s" % e)


def main(agent_name: str) -> None:
    AGENT = DeploymentDRLAgent(agent_name=agent_name, ns=NS_PREFIX)

    try:
        AGENT.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    AGENT_NAME = sys.argv[1]
    main(agent_name=AGENT_NAME)
