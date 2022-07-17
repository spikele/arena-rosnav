#!/usr/bin/env python3
import random

import nav_msgs.srv
import rospkg
import rospy
from map_generator import *
from nav_msgs.msg import OccupancyGrid

from simulator_setup.srv import *


class MapGenerator:
    def __init__(self):
        # initial value for scenario number
        self.nr = -1

        # general map parameter
        self.height = 100
        self.width = 100

        self.resolution = 0.2

        # indoor map parameter
        self.corridor_radius = 6
        self.iterations = 50
        self.room_number = 3
        self.room_width = 10
        self.room_height = 10
        self.no_overlap = True

        # outdoor map parameter
        self.obstacle_number = 15
        self.obstacle_extra_radius = 6

        # initialize occupancy grid
        self.occupancy_grid = OccupancyGrid()
        self.map_type = rospy.get_param("~map_type")
        self.indoor_prob = rospy.get_param("~indoor_prob")

        # self.generate_initial_map() # initial random map generation (before first episode)
        rospy.Subscriber('/map', OccupancyGrid, self.get_occupancy_grid)

        # generate new random map for the next episode when entering new episode
        rospy.Service('/new_map', GetMapWithSeed, self.new_episode_callback)

        self.mappub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)

        # initialize yaml files
        map_dir = os.path.join(rospkg.RosPack().get_path('simulator_setup'), 'maps')
        create_yaml_files('random_map', map_dir, self.resolution)

    # a bit cheating: copy OccupancyGrid meta data from map_server of initial map
    def get_occupancy_grid(self, occgrid_msg):
        self.occupancy_grid = occgrid_msg

    def generate_initial_map(self):  # generate random map png in random_map directory
        map = create_random_map(
            height=self.height,
            width=self.width,
            corridor_radius=self.corridor_radius,
            iterations=self.iterations,
            obstacle_number=self.obstacle_number,
            obstacle_extra_radius=self.obstacle_extra_radius,
            room_number=self.room_number,
            room_width=self.room_width,
            room_height=self.room_height,
            no_overlap=self.no_overlap,
            map_type=self.map_type,
            indoor_prob=self.indoor_prob,
            seed=0
        )
        make_image(map)
        rospy.loginfo("Initial random map generated.")

    def generate_mapdata(self, seed: int = 0):  # generate random map data array for occupancy grid
        map = create_random_map(
            height=self.height,
            width=self.width,
            corridor_radius=self.corridor_radius,
            iterations=self.iterations,
            obstacle_number=self.obstacle_number,
            obstacle_extra_radius=self.obstacle_extra_radius,
            room_number=self.room_number,
            room_width=self.room_width,
            room_height=self.room_height,
            no_overlap=self.no_overlap,
            map_type=self.map_type,
            indoor_prob=self.indoor_prob,
            seed=seed
        )
        make_image(map)
        map = np.flip(map, axis=0)
        # map currently [0,1] 2D np array needs to be flattened for publishing OccupancyGrid.data
        map = (map * 100).flatten()
        return map

    # def new_episode_callback(self,goal_msg: PoseStamped):
    #     current_episode = goal_msg.header.seq
    #     is_new_episode = self.nr != current_episode # self.nr starts with -1 so 0 will be the first new episode
    #     if is_new_episode:
    #         self.nr = current_episode
    #         self.occupancy_grid.data = self.generate_mapdata()
    #         rospy.loginfo("New random map generated for episode {}.".format(self.nr))
    #         self.mappub.publish(self.occupancy_grid)
    #         rospy.loginfo("New random map published.")

    def new_episode_callback(self, request: GetMapWithSeedRequest):
        seed = request.seed
        self.occupancy_grid.data = self.generate_mapdata(seed)
        self.mappub.publish(self.occupancy_grid)
        srv_response = GetMapWithSeedResponse(map=self.occupancy_grid)
        return srv_response


if __name__ == '__main__':
    rospy.init_node('map_generator')
    # if rospy.get_param("map_file") == "random_map":
    task_generator = MapGenerator()
    rospy.spin()

"""#!/usr/bin/env python3

import rospy
from map_generator import *
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import numpy as np
import subprocess

class MapGenerator():
    def __init__(self):
        # initial value for scenario number
        self.nr = -1

        # # general map parameter
        # self.height = rospy.get_param("~map_height")
        # self.width = rospy.get_param("~map_width")

        # # indoor map parameter
        # self.cr = rospy.get_param("~corridor_radius")
        # self.iterations = rospy.get_param("~iterations")

        # # outdoor map parameter
        # self.obsnum = rospy.get_param("~obstacle_number")
        # self.obsrad = rospy.get_param("~obstacle_radius")

       # general map parameter
        self.height = 101
        self.width = 101

        # indoor map parameter
        self.cr = 5
        self.iterations = 100

        # outdoor map parameter
        self.obsnum = 10
        self.obsrad = 1

        # initialize occupancy grid
        self.occupancy_grid = OccupancyGrid()

        # self.generate_initial_map() # initial random map generation (before first episode)
        rospy.Subscriber('/map', OccupancyGrid, self.get_occupancy_grid)
        rospy.Subscriber('/demand', String, self.new_episode_callback) # generate new random map for the next episode when entering new episode
        self.mappub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)

    def get_occupancy_grid(self, occgrid_msg: OccupancyGrid): # a bit cheating: copy OccupancyGrid meta data from map_server of initial map
        self.occupancy_grid = occgrid_msg

    def generate_initial_map(self): # generate random map png in random_map directory
        map = create_random_map(
            height = self.height,
            width = self.width,
            corridor_radius = self.cr,
            iterations = self.iterations,
            obstacle_number = self.obsnum,
            obstacle_extra_radius = self.obsrad
        )
        make_image(map)
        rospy.loginfo("Initial random map generated.")

    def generate_mapdata(self): # generate random map data array for occupancy grid
        map = create_random_map(
            height = self.height,
            width = self.width,
            corridor_radius = self.cr,
            iterations = self.iterations,
            obstacle_number = self.obsnum,
            obstacle_extra_radius = self.obsrad
        )
        make_image(map)
        map = np.flip(map,axis=0)
        map = (map*100).flatten() # map currently [0,1] 2D np array needs to be flattened for publishing OccupancyGrid.data
        return map

    # def new_episode_callback(self,goal_msg: PoseStamped):
    #     current_episode = goal_msg.header.seq
    #     is_new_episode = self.nr != current_episode # self.nr starts with -1 so 0 will be the first new episode
    #     if is_new_episode:
    #         self.nr = current_episode
    #         self.occupancy_grid.data = self.generate_mapdata()
    #         rospy.loginfo("New random map generated for episode {}.".format(self.nr))
    #         self.mappub.publish(self.occupancy_grid)
    #         rospy.loginfo("New random map published.")
    
    def new_episode_callback(self, msg: String):
        self.occupancy_grid.data = self.generate_mapdata()
        # rospy.loginfo("New random map generated for episode {}.".format(self.nr))
        self.mappub.publish(self.occupancy_grid)
        bashCommand = "rosservice call /move_base/clear_costmaps"
        subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        rospy.loginfo("New random map published and costmap cleared.")



if __name__ == '__main__':
    rospy.init_node('map_generator')
    # if rospy.get_param("map_file") == "random_map":
    task_generator = MapGenerator()
    rospy.spin()"""