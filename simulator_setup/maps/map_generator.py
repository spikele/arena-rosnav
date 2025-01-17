import os
import random

import numpy as np
import yaml
from PIL import Image


def create_yaml_files(map_name, dir_path, resolution):
    empty_yaml = {
        "image": "{0}.png".format("empty_map"),
        "resolution": resolution,
        "origin": [0.0, 0.0, 0.0],  # [-x,-y,0.0]
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196}
    map_yaml = {
        "image": "{0}.png".format(map_name),
        "resolution": resolution,
        "origin": [0.0, 0.0, 0.0],  # [-x,-y,0.0]
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196}

    with open(dir_path + "/{0}/map.yaml".format(map_name), 'w') as outfile:
        yaml.dump(map_yaml, outfile, sort_keys=False, default_flow_style=None)
    with open(dir_path + "/{0}/empty.yaml".format(map_name), 'w') as outfile:
        yaml.dump(empty_yaml, outfile, sort_keys=False, default_flow_style=None)

    world_yaml_properties = {"properties": {"velocity_iterations": 10, "position_iterations": 10}}
    world_yaml_layers = {"layers": [
        {"name": "static", "map": "empty.yaml", "color": [0, 1, 0, 1]},
        {"name": "dynamic", "map": "empty.yaml", "color": [0, 1, 0, 1]}
    ]}
    world_layers_properties = {"properties": {"velocity_iterations": 10, "position_iterations": 10}}
    world_layer_layers = {"layers": [
        {"name": "static", "map": "map.yaml", "color": [0, 1, 0, 1]},
        {"name": "dynamic", "map": "map.yaml", "color": [0, 1, 0, 1]}]}

    with open(dir_path + "/{0}/map.world.yaml".format(map_name), 'w') as outfile:
        yaml.dump(world_yaml_properties, outfile, sort_keys=False,
                  default_flow_style=False)  # somehow the first part must be with default_flow_style=False
        yaml.dump(world_yaml_layers, outfile, sort_keys=False,
                  default_flow_style=None)  # 2nd part must be with default_flow_style=None

    with open(dir_path + "/{0}/map_layer.yaml".format(map_name), 'w') as outfile:
        yaml.dump(world_layers_properties, outfile, sort_keys=False,
                  default_flow_style=False)  # somehow the first part must be with default_flow_style=False
        yaml.dump(world_layer_layers, outfile, sort_keys=False,
                  default_flow_style=None)  # 2nd part must be with default_flow_style=None


def make_image(map):  # create PNG file from occupancy map (1:occupied, 0:free) and the necessary yaml files
    img = Image.fromarray(((map - 1) ** 2 * 255).astype('uint8'))  # monochromatic image
    imgrgb = img.convert('RGB')
    map_name = "random_map"
    dir_path = os.path.dirname(os.path.realpath(
        __file__))  # get path for current file, does not work if os.chdir() was used, NOTE: change directory if needed
    try:
        os.mkdir(dir_path + "/" + map_name)  # create directory based on mapname where this script is located
    except:
        pass
    imgrgb.save(dir_path + "/{0}/{0}.png".format(map_name))  # save map in map directory

    create_empty_map(map.shape[0], map.shape[1], map_name, dir_path)


def create_empty_map(height, width, map_name, dir_path):
    map = np.tile(1, [height, width])
    map[slice(1, height - 1), slice(1, width - 1)] = 0
    img = Image.fromarray(((map - 1) ** 2 * 255).astype('uint8'))  # monochromatic image
    imgrgb = img.convert('RGB')
    imgrgb.save(dir_path + "/{0}/{1}.png".format(map_name, "empty_map"))  # save map in map directory


def initialize_map(height, width,
                   type="indoor"):  # create empty map with format given by height,width and initialize empty tree
    if type == "outdoor":
        map = np.tile(1, [height, width])
        map[slice(1, height - 1), slice(1, width - 1)] = 0
        return map
    else:
        return np.tile(1, [height, width])


def insert_root_node(map, tree):  # create root node in center of map
    root_node = [int(np.floor(map.shape[0] / 2)), int(np.floor(map.shape[1] / 2))]
    map[root_node[0], root_node[1]] = 0
    tree.append(root_node)


def sample(map, corridor_radius):  # sample position from map within boundary and leave tolerance for corridor width
    random_x = np.random.choice(range(corridor_radius + 2, map.shape[0] - corridor_radius - 1, 1))
    random_y = np.random.choice(range(corridor_radius + 2, map.shape[1] - corridor_radius - 1, 1))
    return [random_x, random_y]


def find_nearest_node(random_position, tree):  # find nearest node according to L1 norm
    nearest_node = []
    min_distance = np.inf
    for node in tree:
        distance = sum(np.abs(np.array(random_position) - np.array(node)))
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    return nearest_node


def insert_new_node(random_position, tree, map):  # insert new node into the map and tree
    map[random_position[0], random_position[1]] = 0
    tree.append(random_position)


def get_constellation(node1, node2):
    # there are two relevant constellations for the 2 nodes, which must be considered when creating the horizontal and vertical path
    # 1: lower left and upper right
    # 2: upper left and lower right
    # each of the 2 constellation have 2 permutations which must be considered as well
    constellation1 = {
        "permutation1": node1[0] > node2[0] and node1[1] < node2[1],  # x1>x2 and y1<y2
        "permutation2": node1[0] < node2[0] and node1[1] > node2[1]}  # x1<x2 and y1>y2
    if constellation1["permutation1"] or constellation1["permutation2"]:
        return 1
    else:
        return 2


def create_path(node1, node2, corridor_radius, map):
    coin_flip = np.random.random()
    x1, x2 = sorted([node1[0], node2[0]])  # x and y coordinates must be sorted for usage with range function
    y1, y2 = sorted([node1[1], node2[1]])
    if get_constellation(node1, node2) == 1:  # check which constellation
        if coin_flip >= 0.5:  # randomly determine the curvature of the path (right turn/left turn)
            map[slice(x1 - corridor_radius, x1 + corridor_radius + 1), range(y1 - corridor_radius,
                                                                             y2 + 1 + corridor_radius,
                                                                             1)] = 0  # horizontal path
            map[range(x1 - corridor_radius, x2 + 1 + corridor_radius, 1), slice(y1 - corridor_radius,
                                                                                y1 + corridor_radius + 1)] = 0  # vertical path
        else:
            map[slice(x2 - corridor_radius, x2 + corridor_radius + 1), range(y1 - corridor_radius,
                                                                             y2 + 1 + corridor_radius,
                                                                             1)] = 0  # horizontal path
            map[range(x1 - corridor_radius, x2 + 1 + corridor_radius, 1), slice(y2 - corridor_radius,
                                                                                y2 + corridor_radius + 1)] = 0  # vertical path
    else:
        if coin_flip >= 0.5:  # randomly determine the curvature of the path (right turn/left turn)
            map[slice(x1 - corridor_radius, x1 + corridor_radius + 1), range(y1 - corridor_radius,
                                                                             y2 + 1 + corridor_radius,
                                                                             1)] = 0  # horizontal path
            map[range(x1 - corridor_radius, x2 + 1 + corridor_radius, 1), slice(y2 - corridor_radius,
                                                                                y2 + corridor_radius + 1)] = 0  # vertical path
        else:
            map[slice(x2 - corridor_radius, x2 + corridor_radius + 1), range(y1 - corridor_radius,
                                                                             y2 + 1 + corridor_radius,
                                                                             1)] = 0  # horizontal path
            map[range(x1 - corridor_radius, x2 + 1 + corridor_radius, 1), slice(y1 - corridor_radius,
                                                                                y1 + corridor_radius + 1)] = 0  # vertical path


def create_rooms(map, tree, room_number, room_width, room_height, no_overlap):
    rooms_created = 0
    rooms_list = []
    room_vertical_radius = room_width // 2
    room_horizontal_radius = room_height // 2
    distance = 2 * (room_vertical_radius ** 2 + room_horizontal_radius ** 2) ** 0.5
    for room in range(1, room_number + 1):
        if rooms_created == room_number:
            break
        for i, node in enumerate(random.sample(tree, len(tree))):
            x1 = node[0] - room_horizontal_radius
            x2 = node[0] + room_horizontal_radius
            y1 = node[1] - room_vertical_radius
            y2 = node[1] + room_vertical_radius

            if x1 < 0 or x2 > map.shape[1] or y1 < 0 or y2 > map.shape[0]:  # if room is out of map
                if (i == len(tree) - 1) and (room > rooms_created):
                    print("No valid position for room " + str(room) + " found.")
                continue

            if rooms_created == room_number:
                break
            if no_overlap:
                if len(rooms_list) == 0:
                    try:
                        map[x1, y1]
                        map[x1, y2]
                        map[x2, y1]
                        map[x2, y2]
                        map[slice(x1, x2), slice(y1, y2)] = 0
                        rooms_created += 1
                        rooms_list.append(node)
                        print("Room " + str(room) + " created at position " + str(node[0]) + "," + str(node[1]))
                        break
                    except:
                        pass
                else:
                    for room_node in rooms_list:
                        collision = False
                        if np.linalg.norm(np.array(node) - np.array(room_node)) < distance:
                            if (i == len(tree) - 1) and (room > rooms_created):
                                print("No valid position for room " + str(room) + " found.")
                            collision = True
                            break
                    if collision:
                        continue
                    try:
                        map[x1, y1]
                        map[x1, y2]
                        map[x2, y1]
                        map[x2, y2]
                        map[slice(x1, x2), slice(y1, y2)] = 0
                        rooms_created += 1
                        rooms_list.append(node)
                        print("Room " + str(room) + " created at position " + str(node[0]) + "," + str(node[1]))
                        break
                    except:
                        pass

            else:
                try:
                    map[x1, y1]
                    map[x1, y2]
                    map[x2, y1]
                    map[x2, y2]
                    map[slice(x1, x2), slice(y1, y2)] = 0
                    rooms_created += 1
                    rooms_list.append(node)
                    print("Room " + str(room) + " created at position " + str(node[0]) + "," + str(node[1]))
                    break
                except:
                    pass


def create_indoor_map(height, width, corridor_radius, iterations, room_number, room_width, room_height, no_overlap):
    tree = []  # initialize empty tree
    map = initialize_map(height, width)
    insert_root_node(map, tree)
    for i in range(iterations):  # create as many paths/nodes as defined in iteration
        random_position = sample(map, corridor_radius)
        nearest_node = find_nearest_node(random_position,
                                         tree)  # nearest node must be found before inserting the new node into the tree, else nearest node will be itself
        insert_new_node(random_position, tree, map)
        create_path(random_position, nearest_node, corridor_radius, map)
    create_rooms(map, tree, room_number, room_width, room_height, no_overlap)

    # create border around map
    map[0, :] = 1
    map[-1, :] = 1
    map[:, 0] = 1
    map[:, -1] = 1

    return map


def create_outdoor_map(height, width, obstacle_number, obstacle_extra_radius):
    map = initialize_map(height, width, type="outdoor")
    for i in range(obstacle_number):
        random_position = sample(map, obstacle_extra_radius)
        map[slice(random_position[0] - obstacle_extra_radius, random_position[
            0] + obstacle_extra_radius + 1),  # create 1 pixel obstacles with extra radius if specified
            slice(random_position[1] - obstacle_extra_radius, random_position[1] + obstacle_extra_radius + 1)] = 1
    return map


def create_random_map(height, width, corridor_radius, iterations, obstacle_number, obstacle_extra_radius, room_number,
                      room_width, room_height, no_overlap, map_type: str, indoor_prob: float, seed: int):
    np.random.seed(seed)
    random.seed(seed)
    if map_type == "mixed":
        if np.random.random() <= indoor_prob:
            map_type = "indoor"
        else:
            map_type = "outdoor"

    if map_type == "indoor":
        map = create_indoor_map(height, width, corridor_radius, iterations, room_number, room_width, room_height,
                                no_overlap)
        return map
    else:
        map = create_outdoor_map(height, width, obstacle_number, obstacle_extra_radius)
        return map


# NOTE: if you only need the functions comment the main function
if __name__ == '__main__':
    # map generation parameters loaded from config.yaml
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/config.yaml') as f:
        config = yaml.safe_load(f)
    height = config["height"]
    width = config["width"]
    resolution = config["resolution"]
    corridor_radius = config["corridor_radius"]
    iterations = config["iterations"]
    room_number = config["room_number"]
    room_width = config["room_width"]
    room_height = config["room_height"]
    no_overlap = config["no_overlap"]
    obstacle_number = config["obstacle_number"]
    obstacle_extra_radius = config["obstacle_extra_radius"]
    maptype = config["maptype"]
    map_name = config["map_name"]

    # create outdoor map
    if maptype == "outdoor":
        outdoor_map = create_outdoor_map(height, width, obstacle_number, obstacle_extra_radius)
        make_image(outdoor_map, maptype, map_name, resolution)

    # create indoor map
    if maptype == "indoor":
        indoor_map = create_indoor_map(height, width, corridor_radius, iterations, room_number, room_width, room_height,
                                       no_overlap)
        make_image(indoor_map, maptype, map_name, resolution)

"""import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.ticker import FixedLocator
from PIL import Image
import os
import yaml

def create_yaml_files(map_name,dir_path):
    empty_yaml = {
        "image": "{0}.png".format("empty_map"),
        "resolution": 0.20,
        "origin": [0.0,0.0,0.0], # [-x,-y,0.0]
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196}
    map_yaml = {
        "image": "{0}.png".format(map_name),
        "resolution": 0.20,
        "origin": [0.0,0.0,0.0], # [-x,-y,0.0]
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196}
    with open(dir_path+"/{}/map.yaml".format(map_name), 'w') as outfile:
        yaml.dump(map_yaml, outfile, sort_keys=False,default_flow_style=None)
    with open(dir_path+"/{}/empty.yaml".format(map_name), 'w') as outfile:
        yaml.dump(empty_yaml, outfile, sort_keys=False,default_flow_style=None)
    world_yaml_properties = {"properties":{"velocity_iterations": 10,"position_iterations": 10}}
    world_yaml_layers = {"layers":[
        {"name": "static","map": "empty.yaml","color": [0, 1, 0, 1]}#,
        # {"name": "map","map": "map.yaml","color": [0, 0, 1, 1]}
        ]}
    with open(dir_path+"/{}/map.world.yaml".format(map_name), 'w') as outfile:
        yaml.dump(world_yaml_properties, outfile, sort_keys=False,default_flow_style=False) # somehow the first part must be with default_flow_style=False
        yaml.dump(world_yaml_layers, outfile, sort_keys=False,default_flow_style=None) # 2nd part must be with default_flow_style=None

def make_image_for_human(map,map_name): # create human-aestethic PNG file from occupancy map (1:occupied, 0:free)
    height = map.shape[0] 
    width = map.shape[1]
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(map, cmap='binary', interpolation='none') # heatmap, interpolation "none" image scaling big->small, "nearest" small->big
    plt.axis("on")
    ax.axes.get_xaxis().set_visible(True)
    ax.axes.get_yaxis().set_visible(True)   
    ax.set_xticks(range(0,width,10))
    ax.set_xticklabels(range(0,width,10))
    ax.set_yticks(range(0,height,10))
    ax.set_yticklabels(range(0,height,10))
    ax.xaxis.set_minor_locator(FixedLocator(range(0,width,1)))
    ax.yaxis.set_minor_locator(FixedLocator(range(0,height,1)))
    plt.grid("on",which="both",color="black")
    plt.savefig("{0}/{0}_human.png".format(map_name), bbox_inches='tight') 

def make_image(map): # create PNG file from occupancy map (1:occupied, 0:free) and 
    # now = datetime.datetime.now()
    img = Image.fromarray(((map-1)**2*255).astype('uint8')) # monochromatic image
    imgrgb = img.convert('RGB')
    # map_name = "map_{}".format(now.strftime("%Y_%m_%d_%H_%M_%S")) # create mapname from current datetime
    map_name = "random_map"
    dir_path = os.path.dirname(os.path.realpath(__file__)) # get path for current file, does not work if os.chdir() was used
    try:
        os.mkdir(dir_path+"/"+map_name) # create directory based on mapname
    except:
        pass
    imgrgb.save(dir_path+"/{0}/{0}.png".format(map_name)) # save map in map directory
    create_yaml_files(map_name,dir_path) # create corresponding yaml files
    # create empty map with same size as map
    create_empty_map(map.shape[0],map.shape[1],map_name,dir_path)    
    # make_image_for_human(map,map_name) # create human friendly map png
 

def initialize_map(height,width,type="indoor"): # create empty map with format given by height,width and initialize empty tree
    if type == "outdoor":
        map = np.tile(1,[height,width])
        map[slice(1,height-1),slice(1,width-1)] = 0
        return map
    else:
        return np.tile(1,[height,width])

def insert_root_node(map,tree): # create root node in center of map
    root_node = [int(np.floor(map.shape[0]/2)),int(np.floor(map.shape[1]/2))]
    map[root_node[0],root_node[1]] = 0
    tree.append(root_node)

def sample(map,corridor_radius): # sample position from map within boundary and leave tolerance for corridor width
    random_x = np.random.choice(range(corridor_radius+2,map.shape[0]-corridor_radius-1,1))
    random_y = np.random.choice(range(corridor_radius+2,map.shape[1]-corridor_radius-1,1))
    return [random_x,random_y]

def find_nearest_node(random_position,tree): # find nearest node according to L1 norm
    nearest_node = []
    min_distance = np.inf
    for node in tree:
        distance = sum(np.abs(np.array(random_position)-np.array(node)))
        if distance < min_distance:
            min_distance = distance
            nearest_node = node
    return nearest_node

def insert_new_node(random_position,tree,map): # insert new node into the map and tree
    map[random_position[0],random_position[1]] = 0
    tree.append(random_position)

def get_constellation(node1,node2):
    # there are two relevant constellations for the 2 nodes, which must be considered when creating the horizontal and vertical path
    # 1: lower left and upper right
    # 2: upper left and lower right
    # each of the 2 constellation have 2 permutations which must be considered as well    
    constellation1 = {
        "permutation1": node1[0]>node2[0] and node1[1]<node2[1], # x1>x2 and y1<y2
        "permutation2": node1[0]<node2[0] and node1[1]>node2[1]} # x1<x2 and y1>y2
    if constellation1["permutation1"] or constellation1["permutation2"]:
        return 1 
    else:
        return 2

def create_path(node1,node2,corridor_radius,map):
    coin_flip = np.random.random()
    x1,x2 = sorted([node1[0],node2[0]]) # x and y coordinates must be sorted for usage with range function
    y1,y2 = sorted([node1[1],node2[1]])
    if get_constellation(node1,node2)==1: # check which constellation
        if coin_flip>=0.5: # randomly determine the curvature of the path (right turn/left turn)
            map[slice(x1-corridor_radius,x1+corridor_radius+1),range(y1-corridor_radius,y2+1+corridor_radius,1)] = 0 # horizontal path
            map[range(x1-corridor_radius,x2+1+corridor_radius,1),slice(y1-corridor_radius,y1+corridor_radius+1)] = 0 # vertical path
        else:
            map[slice(x2-corridor_radius,x2+corridor_radius+1),range(y1-corridor_radius,y2+1+corridor_radius,1)] = 0 # horizontal path
            map[range(x1-corridor_radius,x2+1+corridor_radius,1),slice(y2-corridor_radius,y2+corridor_radius+1)] = 0 # vertical path
    else:
        if coin_flip>=0.5: # randomly determine the curvature of the path (right turn/left turn)
            map[slice(x1-corridor_radius,x1+corridor_radius+1),range(y1-corridor_radius,y2+1+corridor_radius,1)] = 0 # horizontal path
            map[range(x1-corridor_radius,x2+1+corridor_radius,1),slice(y2-corridor_radius,y2+corridor_radius+1)] = 0 # vertical path
        else:
            map[slice(x2-corridor_radius,x2+corridor_radius+1),range(y1-corridor_radius,y2+1+corridor_radius,1)] = 0 # horizontal path
            map[range(x1-corridor_radius,x2+1+corridor_radius,1),slice(y1-corridor_radius,y1+corridor_radius+1)] = 0 # vertical path

def create_indoor_map(height,width,corridor_radius,iterations):
    tree = [] # initialize empty tree
    map = initialize_map(height,width)
    insert_root_node(map,tree)
    for i in range(iterations): # create as many paths/nodes as defined in iteration
        random_position = sample(map,corridor_radius)
        nearest_node = find_nearest_node(random_position,tree) # nearest node must be found before inserting the new node into the tree, else nearest node will be itself
        insert_new_node(random_position,tree,map)
        create_path(random_position,nearest_node,corridor_radius,map)
    return map

def create_outdoor_map(height,width,obstacle_number,obstacle_extra_radius):
    map = initialize_map(height,width,type="outdoor")
    for i in range(obstacle_number):
        random_position = sample(map,obstacle_extra_radius) 
        map[slice(random_position[0]-obstacle_extra_radius,random_position[0]+obstacle_extra_radius+1), # create 1 pixel obstacles with extra radius if specified
        slice(random_position[1]-obstacle_extra_radius,random_position[1]+obstacle_extra_radius+1)] = 1
    return map

def create_random_map(height,width,corridor_radius,iterations,obstacle_number,obstacle_extra_radius):
    if np.random.random()>=0.5:
        map =  create_indoor_map(height,width,corridor_radius,iterations)
        return map
    else:
        map = create_outdoor_map(height,width,obstacle_number,obstacle_extra_radius)
        return map

def create_empty_map(height,width,map_name,dir_path):
    map = np.tile(1,[height,width])
    map[slice(1,height-1),slice(1,width-1)] = 0
    img = Image.fromarray(((map-1)**2*255).astype('uint8')) # monochromatic image
    imgrgb = img.convert('RGB')
    imgrgb.save(dir_path+"/{0}/{1}.png".format(map_name,"empty_map")) # save map in map directory

# if __name__ == '__main__':
#     make_image(create_empty_map(101,101))"""