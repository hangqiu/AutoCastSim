import collections
from enum import Enum
import math
import numpy as np
from AVR.PCProcess import LidarPreprocessor
import time
from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
import carla
from AVR import Utils


offset = 0
class MotionModel(Enum):
    """
    o-o-o-o-o
    o-------o
    o---X---o
    o-------o
    o-o-o-o-o
    """
    #
    STOP = 0
    #
    STRAIGHT = 1
    #
    LEFTFORWARD30 = 2
    LEFTFORWARD45 = 3
    LEFTFORWARD60 = 4
    #
    LEFT = 5
    #
    LEFTBACKWARD30 = 6
    LEFTBACKWARD45 = 7
    LEFTBACKWARD60 = 8
    #
    RIGHTFORWARD30 = 9
    RIGHTFORWARD45 = 10
    RIGHTFORWARD60 = 11
    #
    RIGHT = 12
    #
    RIGHTBACKWARD30 = 13
    RIGHTBACKWARD45 = 14
    RIGHTBACKWARD60 = 15


class AStarPlanner(object):

    def __init__(self):
        # self.depth_map_min = depth_map_min
        # self.grid_nondrivable = np.zeros(shape=(LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE))
        # self.motion = self.get_motion_model()
        self.ego_Trans = None
        self.debug = False
        # self.planning_start_y = -3
        # MOTION DICT
        self.motion_dict_dx_dy_cost = {
            MotionModel.RIGHT:          [-2, 0, 2],
            MotionModel.RIGHTFORWARD60: [-2, -1, math.sqrt(5)],
            MotionModel.RIGHTFORWARD45: [-2, -2, math.sqrt(8)],
            MotionModel.RIGHTFORWARD30: [-1, -2, math.sqrt(5)],
            MotionModel.STRAIGHT:       [0, -2, 2],
            MotionModel.LEFTFORWARD30:  [1, -2, math.sqrt(5)],
            MotionModel.LEFTFORWARD45:  [2, -2, math.sqrt(8)],
            MotionModel.LEFTFORWARD60:  [2, -1, math.sqrt(5)],
            MotionModel.LEFT:           [2, 0, 2],
        }
        self.next_motion_map = {
            MotionModel.RIGHT:          [MotionModel.RIGHT, MotionModel.RIGHTFORWARD30],
            MotionModel.RIGHTFORWARD60: [MotionModel.RIGHTFORWARD60, MotionModel.RIGHT, MotionModel.RIGHTFORWARD45],
            MotionModel.RIGHTFORWARD45: [MotionModel.RIGHTFORWARD45, MotionModel.RIGHTFORWARD60, MotionModel.RIGHTFORWARD30],
            MotionModel.RIGHTFORWARD30: [MotionModel.RIGHTFORWARD30, MotionModel.RIGHTFORWARD45, MotionModel.STRAIGHT],
            MotionModel.STRAIGHT:       [MotionModel.STRAIGHT, MotionModel.LEFTFORWARD30, MotionModel.RIGHTFORWARD30],
            MotionModel.LEFTFORWARD30:  [MotionModel.LEFTFORWARD30, MotionModel.STRAIGHT, MotionModel.LEFTFORWARD45],
            MotionModel.LEFTFORWARD45:  [MotionModel.LEFTFORWARD45, MotionModel.LEFTFORWARD30, MotionModel.LEFTFORWARD60],
            MotionModel.LEFTFORWARD60:  [MotionModel.LEFTFORWARD60, MotionModel.LEFTFORWARD45, MotionModel.LEFT],
            MotionModel.LEFT:           [MotionModel.LEFT, MotionModel.LEFTFORWARD60],
            MotionModel.STOP:           [MotionModel.LEFT,
                                         MotionModel.LEFTFORWARD60, MotionModel.LEFTFORWARD45, MotionModel.LEFTFORWARD30,
                                         MotionModel.STRAIGHT,
                                         MotionModel.RIGHTFORWARD30, MotionModel.RIGHTFORWARD45, MotionModel.RIGHTFORWARD60,
                                         MotionModel.RIGHT],
        }

    class Node:
        def __init__(self, x, y, cost, prev_idx, last_move=None, deviation_dist=0, lane_type=None):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.prev_idx = prev_idx # previous node index
            self.last_move = last_move
            self.deviation_dist = deviation_dist
            self.lane_type = lane_type

        def __str__(self):
            return "[{} {} {} {} {} {}]".format(self.x, self.y, self.cost, self.prev_idx, self.last_move, self.deviation_dist)

    def planning(self, lidar_proc_res, route_to_destination_loc_tuple_ego_perspective, interpolated_route_to_destination_ego_perspective,
                 grid_thresh, source=(0, 0, 0)):
        start_time = time.time()
        # source[1] += self.planning_start_y
        """ transform to occupancy grid index """
        route_to_destination_nodes = []
        for route_loc_tuple in route_to_destination_loc_tuple_ego_perspective:
            route_x = LidarPreprocessor.getX_grid(route_loc_tuple[0])
            route_y = LidarPreprocessor.getY_grid(route_loc_tuple[1])
            if route_x and route_y:
                route_node = self.Node(route_x, route_y, 0.0, -1)
                route_to_destination_nodes.append(route_node)
                # print("Node: {},{}".format(route_x, route_y))
            else:
                break

        interpolated_route_to_destination_nodes = []
        for route_loc_tuple in interpolated_route_to_destination_ego_perspective:
            route_x = LidarPreprocessor.getX_grid(route_loc_tuple[0])
            route_y = LidarPreprocessor.getY_grid(route_loc_tuple[1])
            if route_x and route_y:
                route_node = self.Node(route_x, route_y, 0.0, -1)
                interpolated_route_to_destination_nodes.append(route_node)
                # print("Node: {},{}".format(route_x, route_y))
            else:
                break

        if self.debug:
            print("================Route to dest nodes ================")
            for node in route_to_destination_nodes:
                print(node)

        self.ego_Trans = CarlaActorPool.get_hero_actor().get_transform()

        source_x = LidarPreprocessor.getX_grid(source[0])
        source_y = LidarPreprocessor.getY_grid(source[1])
        # self.filter_drivable_grids(source_x, source_y, ego_length, ego_width, z_threshold)
        """ iterate thru all intermediate locations """
        rx_m_np, ry_m_np, dist_m_np, deviation_dist_m_np = np.empty(shape=[0]), np.empty(shape=[0]), np.empty(shape=[0]), np.empty(shape=[0])
        source_as_start = False
        complete_path = True
        for i in range(len(route_to_destination_nodes)-1):
            ngoal = route_to_destination_nodes[i+1]
            if not self.verify_node(ngoal, lidar_proc_res, self.ego_Trans, self.debug):
                if self.debug: print("Goal {} not reachable, try next point...".format(ngoal))
                continue

            nstart = route_to_destination_nodes[i]
            if i == 0 or not source_as_start:
                nstart = self.Node(source_x, source_y, 0.0, -1, MotionModel.STOP)
                if abs(nstart.x-ngoal.x) < grid_thresh and abs(nstart.y-ngoal.y) < grid_thresh:
                    continue
                else:
                    source_as_start = True

            nstart.prev_idx = -1 # cuz it's mutable, and changed in last search
            nstart.cost = 0.0
            if nstart.last_move is None:
                nstart.last_move = MotionModel.STOP
            # print("find path")
            if self.debug: print("Search Path {} to {}".format(nstart, ngoal))
            # print("interpolated route ({}):".format(len(interpolated_route_to_destination_nodes)))
            # for n in interpolated_route_to_destination_nodes:
            #     print(str(n))
            # rx, ry, dist, deviation_dist = self.find_path(lidar_proc_res, nstart, ngoal, route_to_destination_nodes[i:i+2], interpolated_route_to_destination_nodes)
            rx, ry, dist, deviation_dist = self.find_path(lidar_proc_res, nstart, ngoal,interpolated_route_to_destination_nodes)

            if len(rx) > 1:
                if self.debug: print("Path found")
                rx_m_np = np.concatenate([rx_m_np, rx])
                # print(rx_m_np.shape)
                ry_m_np = np.concatenate([ry_m_np, ry])
                if dist_m_np.shape[0] != 0:
                    dist += dist_m_np[-1]
                dist_m_np = np.concatenate([dist_m_np, dist])
                deviation_dist_m_np = np.concatenate([deviation_dist_m_np, deviation_dist])
                # print(rx_m_np)
                # print(ry_m_np)
                # print(dist_m_np)
                # print(deviation_dist_m_np)
            # else:
            #     if self.debug: print("Partial path not found")

            if ngoal.prev_idx == -1:
                if self.debug: print("Complete path not found")
                complete_path = False
                break

        return rx_m_np, ry_m_np, dist_m_np, deviation_dist_m_np, complete_path

    def find_path(self, lidar_proc_res, nstart, ngoal, interpolated_route_to_destination_ego_perspective):

        deviation_search_thresh_grid = 10
        farthest_waypoint_index_reached_on_interpolated_route = None

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(nstart)] = nstart

        while 1:

            if len(open_set) == 0:
                break

            # for on in open_set:
            #     # print(on)
            #     if open_set[on].deviation_dist == 0:
            #         open_set[on].deviation_dist = self.calc_interpolated_route_deviation_distance_grid(open_set[on], interpolated_route_to_destination_ego_perspective)


            """ LIFO """
            min_cost = 100000
            c_id = None
            for o in open_set:
                cost = open_set[o].cost + self.calc_heuristic(ngoal, open_set[o]) + open_set[o].deviation_dist * 3 # weight more on the deviation, the weight scalar is a bit hacky here, 3 good for turns,  1 good for straight
                if cost <= min_cost: # LIFO
                    min_cost = cost
                    c_id = o
            # c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(ngoal, open_set[o]) + open_set[o].deviation_dist * 3)  # weight more on the deviation, the weight scalar is a bit hacky here, 3 good for turns,  1 good for straight
            current = open_set[c_id]

            if self.debug:
                # output = []
                # for o in open_set:
                #     output.append("{}: {:.2f}(={:.2f}+{:.2f}+{:.2f})".format(self.calc_XY(o),
                #                                             open_set[o].cost + self.calc_heuristic(ngoal, open_set[o]) +
                #                                             open_set[o].deviation_dist,
                #                                             open_set[o].cost,
                #                                             self.calc_heuristic(ngoal, open_set[o]),
                #                                             open_set[o].deviation_dist))
                # print(output)
                print("Examining {} {}: {:.2f}(={:.2f}+{:.2f}+{:.2f})".format(current.x, current.y,
                                                             current.cost + self.calc_heuristic(ngoal,
                                                                                                current) + current.deviation_dist * 3,
                                                             current.cost,
                                                             self.calc_heuristic(ngoal, current),
                                                             current.deviation_dist * 3))

            goal_range = 1
            if abs(current.x-ngoal.x) <= goal_range and abs(current.y-ngoal.y) <= goal_range:
                # print("Find goal")
                ngoal.prev_idx = current.prev_idx
                ngoal.cost = current.cost
                ngoal.x = current.x
                ngoal.y = current.y
                ngoal.last_move = current.last_move
                ngoal.deviation_dist = current.deviation_dist
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current
            # record the farthest node reached on route
            if current.deviation_dist == 0:
                farthest_waypoint_index_reached_on_interpolated_route = c_id

            # expand_grid search grid based on motion model
            # for i, _ in enumerate(self.motion):
            motion = self.get_motion_model(current.last_move)
            for next_move in motion:
                node = self.Node(current.x + motion[next_move][0],
                                 current.y + motion[next_move][1],
                                 current.cost + motion[next_move][2],
                                 c_id, next_move)
                n_id = self.calc_grid_index(node)

                if n_id in closed_set: # explored before
                    del node
                    continue

                # If the node is not safe, do nothing
                if not self.verify_node(node, lidar_proc_res, self.ego_Trans, self.debug):
                    closed_set[n_id] = node
                    continue

                # """back track surrounding grids and mark invalid (group_assign_grid_value_by_dimension_circle)"""
                # if LidarPreprocessor.is_valid_grid_lanetype(node.x, node.y, self.ego_Trans, carla.LaneType.Sidewalk):
                #     closed_set[n_id] = node
                #     continue

                node.deviation_dist = self.calc_interpolated_route_deviation_distance_grid(node,
                                                                                           interpolated_route_to_destination_ego_perspective)
                if n_id not in open_set:
                    if node.deviation_dist < deviation_search_thresh_grid:
                        open_set[n_id] = node  # discovered a new node
                    else:
                        closed_set[n_id] = node # deviate too far, stop searching
                    # if self.debug: print("Queueing new node {},{}".format(node.x, node.y))
                else:
                    # if self.debug: print("Updating node {},{}".format(node.x, node.y))
                    if open_set[n_id].cost > node.cost: # This path is the best until now. record it
                        del open_set[n_id]
                        open_set[n_id] = node
                    elif open_set[n_id].cost == node.cost:
                        if open_set[n_id].deviation_dist > node.deviation_dist:
                            # break tie by deviation
                            del open_set[n_id]
                            open_set[n_id] = node
                    else:
                        del node

        goal_reached = ngoal
        if ngoal.prev_idx == -1 and farthest_waypoint_index_reached_on_interpolated_route is not None:
            goal_reached = closed_set[farthest_waypoint_index_reached_on_interpolated_route]

        rx, ry, dist, deviation_dist = self.calc_final_path(goal_reached, closed_set)
        # print(rx)
        # print(ry)
        # print("deviation dist ({}): {}".format(len(deviation_dist), deviation_dist))
        rx_m_np = np.array(rx)*LidarPreprocessor.dX + LidarPreprocessor.X_min
        ry_m_np = np.array(ry)*LidarPreprocessor.dY + LidarPreprocessor.Y_min
        dist_m_np = np.array(dist) * LidarPreprocessor.dY # given dX=dY
        deviation_dist_m_np = np.array(deviation_dist) * LidarPreprocessor.dY  # given dX=dY
        # print(rx_np, ry_np)
        return rx_m_np[::-1], ry_m_np[::-1], dist_m_np[::-1],  deviation_dist_m_np[::-1]

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic, was 10
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    @staticmethod
    def calc_interpolated_route_deviation_distance_grid(waypoint_location, interpolated_route_to_destination_nodes):
        if len(interpolated_route_to_destination_nodes) < 2:
            return -1
        x_ego = waypoint_location.x
        y_ego = waypoint_location.y
        # dist_to_dest = math.hypot(x_ego - interpolated_route_to_destination_nodes[-1].x, y_ego - interpolated_route_to_destination_nodes[-1].y)

        # print("ego {}, {}".format(x_ego, y_ego))
        min_distance = None
        for n in interpolated_route_to_destination_nodes:
            # print(n)
            dist = math.hypot(x_ego - n.x, y_ego - n.y)
            # print("node {}, {}".format(n.x, n.y))
            # print("dist {}".format(dist))
            if min_distance is None or dist < min_distance:
                min_distance = dist

        # print(waypoint_location)
        # print(min_distance)
        return min_distance

    @staticmethod
    def calc_route_deviation_distance(waypoint_location, route_to_destination_nodes):
        if len(route_to_destination_nodes) < 2:
            return 0
        x_ego = waypoint_location.x
        y_ego = waypoint_location.y
        dist_to_dest = math.hypot(x_ego-route_to_destination_nodes[-1].x, y_ego-route_to_destination_nodes[-1].y)

        min_distance = None
        for i in range(len(route_to_destination_nodes) - 1):
            route_waypoint_location_src = route_to_destination_nodes[i]
            route_waypoint_location_dst = route_to_destination_nodes[i + 1]
            x_src = route_waypoint_location_src.x
            y_src = route_waypoint_location_src.y
            x_dst = route_waypoint_location_dst.x
            y_dst = route_waypoint_location_dst.y

            dist = abs((x_dst-x_src)*(y_src-y_ego)-(x_src-x_ego)*(y_dst-y_src)) / math.hypot(x_src-x_dst, y_src-y_dst)

            if min_distance is None or dist < min_distance:
                # min_distance = dist
                min_distance = dist * dist_to_dest  # scaling factor, a bit hacky here

        return min_distance

    def get_motion_model(self, last_motion):
        motion = dict()
        if last_motion is None:
            motion[MotionModel.STOP] = self.motion_dict_dx_dy_cost[MotionModel.STOP]
            return motion

        for next_move in self.next_motion_map[last_motion]:
            motion[next_move] = self.motion_dict_dx_dy_cost[next_move]

        return motion

    # @staticmethod
    # def get_motion_model():
    #     motion = [
    #                 [0, -2, 2],  # straight forward
    #                 [1, -2, math.sqrt(5)], [-1, -2, math.sqrt(5)],  # 30 deg forward
    #                 [2, -2, math.sqrt(8)], [-2, -2, math.sqrt(8)], # 45 deg forward
    #                 [2, -1, math.sqrt(5)], [-2, -1, math.sqrt(5)], # 60 deg forward
    #                 [2, 0, 2], [-2, 0, 2],  # 90 deg horizontal
    #              ]
    #     return motion

    def calc_grid_index(self, node):
        return node.y * LidarPreprocessor.X_SIZE + node.x

    def calc_XY(self, grid_index):
        y = int(grid_index / LidarPreprocessor.X_SIZE)
        x = grid_index - y * LidarPreprocessor.X_SIZE
        return [x, y]

    @staticmethod
    def verify_node(node, lidar_proc_res, ego_Trans, debug=False):
        # collision check
        if node.x >= LidarPreprocessor.X_SIZE or node.x < 0:
            if debug: print(node.x, node.y, "x-axis collision")
            return False
        if node.y >= LidarPreprocessor.Y_SIZE or node.y < 0:
            if debug: print(node.x, node.y, "y-axis collision")
            return False
        if lidar_proc_res.obstacle_grid_with_margin_for_planning is not None and lidar_proc_res.obstacle_grid_with_margin_for_planning[node.x][node.y]:
            if debug:print(node.x, node.y, "obstacle grid")
            return False
        # TODO: carla map waypoint lanetype inaccurate, uncomment if improved
        drivable, node.lane_type = LidarPreprocessor.is_valid_grid_lanetype(node.x, node.y, ego_Trans)
        if not drivable:
            if debug: print(node.x, node.y, "not drivable lane type grid: {}".format(node.lane_type))
            return False
        return True

    # def verify_node(self, node):
    #     # collision check
    #     if node.x >= LidarPreprocessor.X_SIZE or node.x < 0:
    #         if self.debug: print(node.x, node.y, "x-axis collision")
    #         return False
    #     if node.y >= LidarPreprocessor.Y_SIZE or node.y < 0:
    #         if self.debug: print(node.x, node.y, "y-axis collision")
    #         return False
    #
    #     if self.grid_nondrivable[node.x][node.y]: # either it is an object or it is null
    #         if self.debug: print(node.x, node.y, "not valid_map")
    #         return False
    #
    #     # drivable space check
    #     node_dist = [node.x * LidarPreprocessor.dX + LidarPreprocessor.X_min, node.y * LidarPreprocessor.dY + LidarPreprocessor.Y_min, 0]
    #     node_loc = Utils.car_to_world_transform(np.array([node_dist]), self.ego_Trans)
    #     node_loc_carla = carla.Location(x=node_loc[0, 0], y=node_loc[0, 1], z=node_loc[0, 2])
    #     node_waypoint = CarlaDataProvider.get_map().get_waypoint(node_loc_carla, lane_type=carla.LaneType.Any)
    #     if node_waypoint.lane_type not in self.valid_lane_type:
    #         if self.debug:
    #             print("None driving lane type {} at {}, ({}) + ({})".format(node_waypoint.lane_type, node_waypoint.transform.location,
    #                                                                     self.ego_Trans.location,
    #                                                                     node_dist))
    #         return False
    #
    #     return True

    def calc_final_path(self, ngoal, closedset):
        # generate final course
        rx, ry = [ngoal.x], [ngoal.y]
        cost = [ngoal.cost]
        deviation = [ngoal.deviation_dist]
        pind = ngoal.prev_idx
        while pind != -1:
            n = closedset[pind]
            rx.append(n.x)
            ry.append(n.y)
            cost.append(n.cost)
            deviation.append(n.deviation_dist)
            pind = n.prev_idx
        return rx, ry, cost, deviation

    # def filter_drivable_grids(self, source_x, source_y, ego_length, ego_width, z_threshold):
    #     z_threshold_mask = np.array(self.depth_map_min <= z_threshold)
    #     for i in range(z_threshold_mask.shape[0]):
    #         for j in range(z_threshold_mask.shape[1]):
    #             if z_threshold_mask[i, j]:
    #                 self.group_assign_grid_value_by_dimension_block(i, j, ego_length, ego_width, 1)
    #     # clear around ego car, drivable
    #     self.group_assign_grid_value_by_dimension_block(source_x, source_y, ego_length, ego_width, 0)
    #
    # def filter_drivable_grids(self, source_x, source_y, ego_length, ego_width, z_threshold):
    #     # Assume dX = dY = dZ
    #     # z_threshold = 0.5
    #     B = max(ego_length / LidarPreprocessor.dX, ego_width / LidarPreprocessor.dX)
    #     for i in self.depth_map_min.keys():
    #         # if source_x - offset <= i <= dest_x + offset or dest_x - offset <= i <= source_x + offset:
    #         for j in self.depth_map_min[i].keys():
    #             # if source_y - offset <= j <= dest_y + offset or dest_y - offset <= j <= source_y + offset:
    #             if self.depth_map_min[i][j] <= z_threshold:
    #                 # self.valid_map[i][j] = 1
    #                 # IsValidGridWayPointWithEnoughSpaceForCar
    #                 x_min = int(max(0, i - B / 2 - offset))
    #                 x_max = int(min(LidarPreprocessor.X_SIZE, i + B / 2 + offset))
    #                 y_min = int(max(0, j - B / 2 - offset))
    #                 y_max = int(min(LidarPreprocessor.Y_SIZE, j + B / 2 + offset))
    #                 # for x in range(x_min, x_max):
    #                 #     self.valid_map[x][j] = 1
    #                 # for y in range(y_min, y_max):
    #                 #     self.valid_map[i][y] = 1
    #                 for x in range(x_min, x_max):
    #                     for y in range(y_min, y_max):
    #                         self.grid_nondrivable[x][y] = 1
    #                 # print(i, j)
    #     self.purge_map_in_ego_boundingbox(source_x, source_y, ego_length, ego_width)

    # def group_assign_grid_value_by_dimension_block(self, i, j, length, width, value):
    #     B = max(length/LidarPreprocessor.dY, width/LidarPreprocessor.dX)
    #     x_min = int(max(0, i - B / 2 - offset))
    #     x_max = int(min(LidarPreprocessor.X_SIZE, i + B / 2 + offset))
    #     y_min = int(max(0, j - B / 2 - offset))
    #     y_max = int(min(LidarPreprocessor.Y_SIZE, j + B / 2 + offset))
    #     for x in range(x_min, x_max):
    #         for y in range(y_min, y_max):
    #             self.grid_nondrivable[x][y] = value

