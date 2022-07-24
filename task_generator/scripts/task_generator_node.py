#! /usr/bin/env python3

import rospy
import rosservice
import subprocess
import time
from std_srvs.srv import EmptyResponse
from nav_msgs.msg import Odometry
from task_generator.tasks import get_predefined_task
from std_msgs.msg import Int16, Bool
# for clearing costmap
from clear_costmap import clear_costmaps
from simulator_setup.srv import * # GetMapWithSeedRequest srv
from pathlib import Path
import json

class TaskGenerator:
    def __init__(self):
        #
        self.sr = rospy.Publisher('/scenario_reset', Int16, queue_size=1)
        self.nr = 0
        self.mode = rospy.get_param("~task_mode")
        
        
        scenarios_json_path = rospy.get_param("~scenarios_json_path")
       
        paths = {"scenario": scenarios_json_path}
  
        self.task = get_predefined_task("",self.mode, PATHS=paths)
       
        # random eval init
        if self.mode == "random_eval":
            # load random eval params
            json_path = Path(paths["scenario"])
            assert json_path.is_file() and json_path.suffix == ".json"
            random_eval_config = json.load(json_path.open())
            self.seed = random_eval_config["seed"]
            self.random_eval_repeats = random_eval_config["repeats"]

            # set up get random map service
            self.request_new_map = rospy.ServiceProxy('/new_map', GetMapWithSeed)  
            # self.request_new_map = rospy.ServiceProxy("/" + self.ns + "/new_map", GetMapWithSeed)
            service_name = '/new_map'
            service_list = rosservice.get_service_list()
            max_tries = 10
            for i in range(max_tries):
                if service_name in service_list:
                    break
                else:
                    time.sleep(1)

        if self.mode == "random_eval_one_map":
            # load random eval params
            json_path = Path(paths["scenario"])
            assert json_path.is_file() and json_path.suffix == ".json"
            random_eval_config = json.load(json_path.open())
            self.seed = random_eval_config["seed"]
            self.random_eval_repeats = random_eval_config["repeats"]

        # if auto_reset is set to true, the task generator will automatically reset the task
        # this can be activated only when the mode set to 'ScenarioTask'
        auto_reset = rospy.get_param("~auto_reset")

        # if the distance between the robot and goal_pos is smaller than this value, task will be reset
        self.timeout_= rospy.get_param("~timeout")
        self.timeout_= self.timeout_*60             # sec
        self.start_time_=rospy.get_time()           # sec
        self.delta_ = rospy.get_param("~delta")
        robot_odom_topic_name = rospy.get_param(
            "robot_odom_topic_name", "odom")
        
        auto_reset = auto_reset and self.mode in ["scenario","random_eval","random_eval_one_map"]
        self.curr_goal_pos_ = None
        
        self.pub = rospy.Publisher('End_of_scenario', Bool, queue_size=10)
        
        if auto_reset:

            rospy.loginfo(
                "Task Generator is set to auto_reset mode, Task will be automatically reset as the robot approaching the goal_pos")
            self.reset_task()
            self.robot_pos_sub_ = rospy.Subscriber(
                robot_odom_topic_name, Odometry, self.check_robot_pos_callback)

            rospy.Timer(rospy.Duration(0.5),self.goal_reached)
            
        else:
            # declare new service task_generator, request are handled in callback task generate
            self.reset_task()
            # self.task_generator_srv_ = rospy.Service(
            #     'task_generator', Empty, self.reset_srv_callback)
                
        self.err_g = 100
        


    def goal_reached(self,event):

        if self.err_g < self.delta_:
            print(self.err_g)
            self.reset_task()
        if rospy.get_time()-self.start_time_>self.timeout_:
            print("timeout")
            self.reset_task()


    # def clear_costmaps(self):
    #     bashCommand = "rosservice call /move_base/clear_costmaps"
    #     process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    #     output, error = process.communicate()
    #     print([output, error])

    def reset_srv_callback(self, req):
        rospy.loginfo("Task Generator received task-reset request!")
        self.task.reset()
        return EmptyResponse()


    def reset_task(self):
        self.start_time_=rospy.get_time()
        if self.mode == "random_eval":
            if self.random_eval_repeats >= self.nr: 
                # change seed per current episode
                seed = (self.random_eval_repeats * (1 + self.nr)) * self.seed 
                # get new map from map generator node
                self.new_map = self.request_new_map(seed)
                # reset task and hand over new map for map update and seed for the task generation
                info = self.task.reset(self.new_map,seed)
            else: # self termination
                subprocess.call(["killall","-9","rosmaster"]) # apt-get install psmisc necessary
                sys.exit()
        elif self.mode == "random_eval_one_map":
            if self.random_eval_repeats >= self.nr: 
                # reset task and hand over seed for the task generation
                seed = (self.random_eval_repeats * (1 + self.nr)) * self.seed 
                info = self.task.reset(seed)
            else: # self termination
                subprocess.call(["killall","-9","rosmaster"]) # apt-get install psmisc necessary
                sys.exit()
        else:
            info = self.task.reset()
        clear_costmaps()
        if info is not None:
            if info == "End":
                print(info)
                # communicates to launch_arena (if used) the end of the simulation
                print("SENDING END MESSAGE")
                self.end_msg = Bool()
                self.end_msg.data = True
                self.pub.publish(self.end_msg)
                sys.exit()
            else:
                self.curr_goal_pos_ = info['robot_goal_pos']
        rospy.loginfo("".join(["="]*80))
        rospy.loginfo("goal reached and task reset!")
        rospy.loginfo("".join(["="]*80))
        self.sr.publish(self.nr)
        self.nr += 1

    def check_robot_pos_callback(self, odom_msg: Odometry):
        robot_pos = odom_msg.pose.pose.position
        robot_x = robot_pos.x
        robot_y = robot_pos.y
        goal_x = self.curr_goal_pos_[0]
        goal_y = self.curr_goal_pos_[1]

        self.err_g = (robot_x-goal_x)**2+(robot_y-goal_y)**2
           


if __name__ == '__main__':
    rospy.init_node('task_generator')
    task_generator = TaskGenerator()
    rospy.spin()




"""#! /usr/bin/env python3

import rospy
import subprocess
from std_srvs.srv import EmptyResponse
from nav_msgs.msg import Odometry
from task_generator.tasks import get_predefined_task
from std_msgs.msg import Int16, Bool
# for clearing costmap
from clear_costmap import clear_costmaps
class TaskGenerator:
    def __init__(self):
        #
        self.sr = rospy.Publisher('/scenario_reset', Int16, queue_size=1)
        self.nr = 0
        mode = rospy.get_param("~task_mode")
        
        
        scenarios_json_path = rospy.get_param("~scenarios_json_path")
       
        paths = {"scenario": scenarios_json_path}
  
        self.task = get_predefined_task("",mode, PATHS=paths)
       


        # if auto_reset is set to true, the task generator will automatically reset the task
        # this can be activated only when the mode set to 'ScenarioTask'
        auto_reset = rospy.get_param("~auto_reset")

        # if the distance between the robot and goal_pos is smaller than this value, task will be reset
        self.timeout_= rospy.get_param("~timeout")
        self.timeout_= self.timeout_*60             # sec
        self.start_time_=rospy.get_time()           # sec
        self.delta_ = rospy.get_param("~delta")
        robot_odom_topic_name = rospy.get_param(
            "robot_odom_topic_name", "odom")
        
        auto_reset = auto_reset and (mode == "scenario" or mode == "random")# NEW AND CHANGED
        # and mode == "scenario" # NEW AND CHANGED
        self.curr_goal_pos_ = None
        
        self.pub = rospy.Publisher('End_of_scenario', Bool, queue_size=10)
        
        if auto_reset:

            rospy.loginfo(
                "Task Generator is set to auto_reset mode, Task will be automatically reset as the robot approaching the goal_pos")
            self.reset_task()
            self.robot_pos_sub_ = rospy.Subscriber(
                robot_odom_topic_name, Odometry, self.check_robot_pos_callback)

            rospy.Timer(rospy.Duration(0.5),self.goal_reached)
            
        else:
            # declare new service task_generator, request are handled in callback task generate
            self.reset_task()
            # self.task_generator_srv_ = rospy.Service(
            #     'task_generator', Empty, self.reset_srv_callback)
                
        self.err_g = 100
        


    def goal_reached(self,event):

        if self.err_g < self.delta_:
            print(self.err_g)
            self.reset_task()
        if rospy.get_time()-self.start_time_>self.timeout_:
            print("timeout")
            self.reset_task()


    # def clear_costmaps(self):
    #     bashCommand = "rosservice call /move_base/clear_costmaps"
    #     process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    #     output, error = process.communicate()
    #     print([output, error])

    def reset_srv_callback(self, req):
        rospy.loginfo("Task Generator received task-reset request!")
        self.task.reset()
        return EmptyResponse()


    def reset_task(self):
        self.start_time_=rospy.get_time()
        info = self.task.reset()
        # clear_costmaps()
        if info is not None:
            if info == "End":
                print(info)
                # communicates to launch_arena (if used) the end of the simulation
                print("SENDING END MESSAGE")
                self.end_msg = Bool()
                self.end_msg.data = True
                self.pub.publish(self.end_msg)
                rospy.signal_shutdown("Finished all episodes of the current scenario")
            else:
                self.curr_goal_pos_ = info['robot_goal_pos']
        rospy.loginfo("".join(["="]*80))
        rospy.loginfo("testnjakslnkdjsfnakjdlnslafjldsanfkjsdanfkldsnfkj" + str(info)) # NEW
        rospy.loginfo("goal reached and task reset!")
        rospy.loginfo("".join(["="]*80))
        self.sr.publish(self.nr)
        self.nr += 1

    def check_robot_pos_callback(self, odom_msg: Odometry):
        robot_pos = odom_msg.pose.pose.position
        robot_x = robot_pos.x
        robot_y = robot_pos.y
        goal_x = self.curr_goal_pos_[0]
        goal_y = self.curr_goal_pos_[1]

        self.err_g = (robot_x-goal_x)**2+(robot_y-goal_y)**2
           


if __name__ == '__main__':
    rospy.init_node('task_generator')
    task_generator = TaskGenerator()
    rospy.spin()"""