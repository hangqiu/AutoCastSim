#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """
import copy
import os

import numpy as np
import time
import carla
import imageio
from agents.navigation.agent import Agent, AgentState
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.tools.misc import is_within_distance_ahead, compute_magnitude_angle
from srunner.tools.route_manipulation import interpolate_trajectory

from AVR import Utils
from AVR import PCProcess
from AVR.autocast_agents.local_planner import LocalPlanner
from AVR.autocast_agents import PathPlanner
from AVR.autocast_agents import AStarPlanner
from AVR.autocast_agents.AStarPlanner import AStarPlanner
from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
from AVR import Collaborator
from collections import defaultdict
import math
from AVR.PCProcess import LidarPreprocessor

grid_thresh = 7  # grids
target_location_evict_grid_range_thresh = 3.5  # meter


class NewAgent(Agent):
    """
    NewAgent is a clone of the basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(NewAgent, self).__init__(vehicle)

        self.id = vehicle.id
        self._proximity_threshold = 10.0  # meters was 10.0
        self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.4,
            'K_I': 0,
            'dt': 1.0 / 20.0}
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed': target_speed,
                                     'lateral_control_dict': args_lateral_dict,
                                     'max_brake': 0.75,
                                     'max_throttle': 0.75
                                     })
        self._hop_resolution = 2.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._grp = None

        # Path planning
        self.ego_trajectory_points_timestamp = []
        self.ego_trajectory_waypoint_distance = []
        self.ego_trajectory_deviation_distance = []
        self.collider_trajectory_points_timestamp = []
        self.filtered_object_list = []

        self.collider_objects_list = []
        self.collider_id_list = []
        self.collider_pkt_loss_count_list = []

        self.collision_detected = False
        self.trajectory_clear = True
        self.collision_detail = None
        # self.last_waypoints = []
        # self.count = 0
        # self.count2 = Utils.AutoPeriod*2
        # self.count3 = 0
        # self.last_destination = [-1.0, -1.0, -1.0]
        self.look_ahaed_waypoints = 2
        self.control_route_loc_tuple_buffer = []  # """should contain last waypoint and look_ahead number of waypoints ahead"""
        self.interpolated_control_route_buffer = []
        self.drawing_object_list = []
        # self.original_mode = True
        self.planner_success = False
        self.complete_path = True
        self.last_control = None

        self.debug_dir = None

        self.next_location = None
        if Utils.AgentDebug:
            self.debug_dir = "{}/{}/Debug/".format(Utils.RecordingOutput, str(Utils.EvalEnv.get_trace_id()))
            if not os.path.exists(self.debug_dir):
                os.mkdir(self.debug_dir)

    def set_destination(self, location):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2]))

        route_trace = self._trace_route(start_waypoint, end_waypoint)
        assert route_trace

        self._local_planner.set_global_plan(route_trace)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def update_ego_trajectory_for_reuse(self, ego_transform, frameId, ego_speed):
        """ pop past waypoints"""
        max_index = -1
        for i in range(len(self.ego_trajectory_points_timestamp)):
            wp = self.ego_trajectory_points_timestamp[i]
            dist = math.hypot(ego_transform.location.x - wp[0], ego_transform.location.y - wp[1])
            if dist < LidarPreprocessor.dX:
                max_index = i
                break
        for j in range(max_index):
            self.ego_trajectory_points_timestamp.pop(0)
            self.ego_trajectory_deviation_distance.pop(0)
            self.ego_trajectory_waypoint_distance.pop(0)
        prev_dist = self.ego_trajectory_waypoint_distance[0]
        for i in range(len(self.ego_trajectory_waypoint_distance)):
            self.ego_trajectory_waypoint_distance[i] = self.ego_trajectory_waypoint_distance[i] - \
                                                       prev_dist

        """ recalc timestamp """
        # temp_traj = np.array(self.ego_trajectory_points_timestamp)
        # print(temp_traj[..., 3])
        # temp_traj[..., 3] -= temp_traj[0, 3]
        # temp_traj[..., 3] += frameId
        # print(temp_traj[..., 3])
        # self.ego_trajectory_points_timestamp = temp_traj.tolist()

        ego_spd_val = math.hypot(ego_speed.x, ego_speed.y)
        framestamp = PathPlanner.estimate_trajectory_framestamp(frameId, ego_spd_val,
                                                                np.array(self.ego_trajectory_waypoint_distance))
        temp_traj = np.array(self.ego_trajectory_points_timestamp)
        # print(temp_traj[..., 3])
        temp_traj[..., 3] = framestamp
        # print(framestamp)
        self.ego_trajectory_points_timestamp = temp_traj.tolist()

    def check_ego_trajectory_validity(self, lidar_proc_res, ego_transform):
        # print("Reuse check:")
        for i, wp in enumerate(self.ego_trajectory_points_timestamp):
            xyz = Utils.world_to_car_transform(np.array([wp[0:3]]), ego_transform).tolist()[0]
            x_grid = LidarPreprocessor.getX_grid(xyz[0])
            y_grid = LidarPreprocessor.getY_grid(xyz[1])
            if x_grid is None or y_grid is None:
                return False
            node = AStarPlanner.Node(x_grid, y_grid, 0.0, -1)
            res = AStarPlanner.verify_node(node, lidar_proc_res, ego_transform)
            if not res:
                # print("Reuse check: ===== Node {} ({},{}), ({},{}) not valid".format(i, x_grid, y_grid, wp[0], wp[1]))
                return False
        return True

    def run_control(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        minimum_ego_trajectory_length = 8
        time_start = time.time()

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        walker_list = actor_list.filter("*walker*")
        lights_list = actor_list.filter("*traffic_light*")
        hazard_detected = self.get_agent_hazard_state(vehicle_list, walker_list, lights_list)

        """ Sensor Processing """
        fused_sensor_id = str(self.id) + Collaborator.FusedLidarSensorName
        frameId = input_data[fused_sensor_id][0]
        fused_sensor_data_pcframe = input_data[fused_sensor_id][1]

        hero_actor = CarlaActorPool.get_hero_actor()
        ego_transform = hero_actor.get_transform()
        ego_speed = hero_actor.get_velocity()

        if Utils.Fast_Lidar:
            lidar_proc_res = LidarPreprocessor.process_lidar_fast(fused_sensor_data_pcframe.pc,
                                                                  fused_sensor_data_pcframe.trans,
                                                                  z_threshold=hero_actor.bounding_box.extent.z * 2,
                                                                  ego_length=self._vehicle.bounding_box.extent.x * 2,
                                                                  ego_width=self._vehicle.bounding_box.extent.y * 2)
        else:
            lidar_proc_res = LidarPreprocessor.process_lidar(fused_sensor_data_pcframe.pc,
                                                             fused_sensor_data_pcframe.trans,
                                                             z_threshold=hero_actor.bounding_box.extent.z * 2,
                                                             ego_actor=self._vehicle,
                                                             ego_length=self._vehicle.bounding_box.extent.x * 2,
                                                             ego_width=self._vehicle.bounding_box.extent.y * 2)

        if Utils.AgentDebug:
            outputdir = self.debug_dir + "merged_obstacle_grid_{}.png".format(frameId)
            LidarPreprocessor.save_binary_occupancy_grid(outputdir, lidar_proc_res.obstacle_grid)
            outputdir = self.debug_dir + "merged_obstacle_grid_with_margin_{}.png".format(frameId)
            LidarPreprocessor.save_binary_occupancy_grid(outputdir,
                                                         lidar_proc_res.obstacle_grid_with_margin_for_planning)
            outputdir = self.debug_dir + "merged_filtered_actor_grid_{}.png".format(frameId)
            LidarPreprocessor.save_binary_occupancy_grid(outputdir, lidar_proc_res.filtered_actor_grid)

        lidar_proc = time.time()
        if Utils.TIMEPROFILE: print("\t\tLidar Processing: {} s".format(lidar_proc - time_start))

        """ Path Planning """
        if len(self.control_route_loc_tuple_buffer) == 0:
            ego_loc_tuple = (ego_transform.location.x, ego_transform.location.y, ego_transform.location.z)
            self.control_route_loc_tuple_buffer.append(
                ego_loc_tuple)  # """init with starting location as first waypoint"""

        self._local_planner.run_step()
        change, waypoint_loc_tuple_from_buffer = self.update_short_horizon_buffer()

        """ planner trajectory infrequent update: check if previous path is valid, then reuse """
        validity = False
        if Utils.PATH_REUSE and self.planner_success and (not change) and len(
                self.ego_trajectory_points_timestamp) > minimum_ego_trajectory_length:
            self.update_ego_trajectory_for_reuse(ego_transform, frameId, ego_speed)
            validity = self.check_ego_trajectory_validity(lidar_proc_res, ego_transform)

        if not validity or len(self.ego_trajectory_points_timestamp) <= minimum_ego_trajectory_length:
            [planned_trajectory, waypoints_dist, waypoints_deviation_dist, self.planner_success,
             self.complete_path] = PathPlanner.plan_trajectory_with_timestamp_astar(
                lidar_proc_res,
                grid_thresh,
                self.control_route_loc_tuple_buffer,
                self.interpolated_control_route_buffer,
                frameId,
                ego_transform,
                ego_speed)

            if not self.planner_success:
                # print("No route, defaulting to previous routes...")
                pass
            else:
                self.ego_trajectory_points_timestamp = planned_trajectory
                self.ego_trajectory_waypoint_distance = waypoints_dist
                self.ego_trajectory_deviation_distance = waypoints_deviation_dist

        else:
            print("Reusing previous path")

        """Status Update"""
        print("Frame ID {}, Ego ID {}".format(frameId, self.id))
        if not Utils.TIMEPROFILE:
            print("Ego Location: {}".format(ego_transform.location))
            print("Ego Rotation: {}".format(ego_transform.rotation))
            print("Route waypoint buffer: {}".format(waypoint_loc_tuple_from_buffer))
            print("Destination Route: {}".format(self.control_route_loc_tuple_buffer))
            print("Trajectory ({}): ".format(len(self.ego_trajectory_points_timestamp)))
            print("Waypoint Distance ({}): ".format(len(self.ego_trajectory_waypoint_distance)))
            # print(self.ego_trajectory_waypoint_distance)
            print("Deviation ({}): ".format(len(self.ego_trajectory_waypoint_distance)))

        path_time = time.time()
        if Utils.TIMEPROFILE: print("\t\tPath Planning: {} s".format(path_time - lidar_proc))

        if not Utils.TIMEPROFILE: print("Planner Success: " + str(self.planner_success))

        """ Control """
        """ control is also focusing on a sliding target, since nearby waypoints will be evicted"""
        # drive_and_waypoint_update
        ego_trajectory_for_collision = copy.deepcopy(self.ego_trajectory_points_timestamp)
        ego_deviation_for_collision = self.ego_trajectory_deviation_distance
        if not self.planner_success:  # for incomplete path only, shorter than 5 meter planning is recognized as no route ahead
            control = self.stop_and_waypoint_timestamp_update()
            ego_trajectory_for_collision = []
            ego_deviation_for_collision = []
        else:
            self.next_location = Utils.evict_till_outside_eviction_range(ego_transform.location,
                                                                         self.ego_trajectory_points_timestamp,
                                                                         target_location_evict_grid_range_thresh)
            control = self._local_planner.run_step(goal=self.next_location)

            if not Utils.TIMEPROFILE: print("Next_Location:{}".format(self.next_location))
        if not Utils.TIMEPROFILE: print(
            "Cur Control: Throttle: {}, Steer: {}, Brake: {}, ".format(control.throttle, control.steer, control.brake))

        control_time = time.time()
        if Utils.TIMEPROFILE: print("\t\tControl: {} s".format(control_time - path_time))

        if Utils.Extrapolation:
            [self.collider_trajectory_points_timestamp, self.collision_detected, self.trajectory_clear,
             self.collision_detail] = \
                self.collider_trajectory_prediction_with_extrapolation(lidar_proc_res.detected_object_list,
                                                                       vehicle_list,
                                                                       # TODO vehicle list for now.. can include peds later
                                                                       ego_trajectory_for_collision,
                                                                       ego_deviation_for_collision)
        else:
            [self.collider_trajectory_points_timestamp, self.collision_detected, self.trajectory_clear,
             self.collision_detail] = \
                self.collider_trajectory_prediction(lidar_proc_res.detected_object_list, vehicle_list,
                                                    # TODO vehicle list for now.. can include peds later
                                                    ego_trajectory_for_collision, ego_deviation_for_collision)

        if not Utils.TIMEPROFILE: print("Trajectory Clear: " + str(self.trajectory_clear))

        if not self.trajectory_clear or not Utils.InTriggerRegion_GlobalUtilFlag:
            control = self.stop_and_waypoint_timestamp_update()

        collider_time = time.time()
        if Utils.TIMEPROFILE: print("\t\tCollision Avoidance: {} s".format(collider_time - path_time))

        # """filter collider trajectory from current timestamp"""
        # if self.collider_trajectory_points_timestamp is not None and len(self.collider_trajectory_points_timestamp) != 0:
        #     self.collider_trajectory_points_timestamp = np.array(self.collider_trajectory_points_timestamp)
        #     self.collider_trajectory_points_timestamp = self.collider_trajectory_points_timestamp[self.collider_trajectory_points_timestamp[..., 3] > frameId]
        #     self.collider_trajectory_points_timestamp = self.collider_trajectory_points_timestamp.tolist()

        """ handling dynamic object list"""
        self.drawing_object_list = PCProcess.extract_actor_bbox_ego_perspective(vehicle_list)

        result_vector = [ego_trajectory_for_collision, self.collider_trajectory_points_timestamp,
                         self.drawing_object_list, self.next_location, 0]

        self.last_control = control
        # print("Controller done!!!")
        return control, result_vector

    def get_agent_hazard_state(self, vehicle_list, walker_list, lights_list):
        # check possible obstacles
        hazard_detected = False
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        walker_state, walker = self._is_vehicle_hazard(walker_list)
        if vehicle_state:
            # print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))
            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

            #### judge if the vehicle ahead is also blocked or dead
            vehicle_state_ahead, _ = self._is_vehicle_ahead_in_hazard(vehicle_list, vehicle)
            # walker_state_ahead, _ = self._is_vehicle_ahead_in_hazard(walker_list, vehicle)
            # if not vehicle_state_ahead and not walker_state_ahead:
            # TODO: fix this
            if not vehicle_state_ahead:
                self._state = 4

        if walker_state:
            # if debug:
            # print('!!! WALKER BLOCKING AHEAD [{}])'.format(walker.id))
            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            # print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))
            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True
        return hazard_detected

    def _is_vehicle_ahead_in_hazard(self, vehicle_list, vehicle_ahead):

        ego_vehicle_location = vehicle_ahead.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            if target_vehicle.id == vehicle_ahead.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            loc = target_vehicle.get_location()

            bumper_to_bumper_dist = (target_vehicle.bounding_box.extent.x - target_vehicle.bounding_box.location.x) \
                                    + (
                                                vehicle_ahead.bounding_box.extent.x + vehicle_ahead.bounding_box.location.x)  # add extra space for extent
            # if is_within_distance_ahead(loc, ego_vehicle_location,
            #                             vehicle_ahead.get_transform().rotation.yaw,
            #                             self._proximity_threshold + bumper_to_bumper_dist):
            if is_within_distance_ahead(target_vehicle.get_transform(),
                                        vehicle_ahead.get_transform(),
                                        self._proximity_threshold + bumper_to_bumper_dist):
                return (True, target_vehicle)

            for i in range(1, 5):
                future_loc = target_vehicle_waypoint.next(i)[0].transform.location
                # if is_within_distance_ahead(future_loc, ego_vehicle_location,
                #                             vehicle_ahead.get_transform().rotation.yaw,
                #                             self._proximity_threshold + bumper_to_bumper_dist):
                if is_within_distance_ahead(target_vehicle_waypoint.next(i)[0].transform,
                                            vehicle_ahead.get_transform(),
                                            self._proximity_threshold + bumper_to_bumper_dist):
                    return (True, target_vehicle)

        return (False, None)

    def stop_and_waypoint_timestamp_update(self):
        control = Utils.stop_control()
        control.brake = self._local_planner._max_brake
        # if self.last_control is not None:
        #     control = self.last_control
        #     control.throttle = 0.0
        #     control.brake = self._local_planner._max_brake

        # TODO: What is below doing? error: waypoint is empty pf size 0? comment out for now
        if len(self.ego_trajectory_points_timestamp) != 0:
            self.ego_trajectory_points_timestamp = np.array(self.ego_trajectory_points_timestamp)
            self.ego_trajectory_points_timestamp[..., 3] += 1
            self.ego_trajectory_points_timestamp = self.ego_trajectory_points_timestamp.tolist()
        return control

    # def drive_and_waypoint_update(self, destination):
    #     next_location = destination
    #     if len(self.trajectory_points_timestamp) != 0:
    #         next_location = self.trajectory_points_timestamp[0]
    #         self.first_point_outside_eviction_range()
    #     else:
    #         self.original_mode = True
    #     control = self._local_planner.run_step(goal=next_location, debug=False)
    #     print("Next_Location:{}".format(next_location))
    #     return control

    def update_short_horizon_buffer(self):
        ''' use the number of look_ahaed_waypoints as a sliding window planner,
                since local planner will evict the first one when close enough'''
        waypoint_loc_tuple_from_buffer = []
        i = 0
        if Utils.AgentDebug: print("Local Planner Waypoint Buffer")
        for wp, _ in self._local_planner._waypoint_buffer:
            loc = (wp.transform.location.x, wp.transform.location.y, wp.transform.location.z)
            if Utils.AgentDebug: print(loc)
            waypoint_loc_tuple_from_buffer.append(loc)
            i += 1
            if i == self.look_ahaed_waypoints:
                break

        i = 0
        change = False
        for waypoint_loc_tuple in waypoint_loc_tuple_from_buffer:
            found = False
            while i < len(self.control_route_loc_tuple_buffer):
                if waypoint_loc_tuple[0] == self.control_route_loc_tuple_buffer[i][0] \
                        and waypoint_loc_tuple[1] == self.control_route_loc_tuple_buffer[i][1]:
                    i += 1
                    found = True
                    break
                i += 1
            if found:
                """ evict all but 2 before this waypoint """
                backoff = 3
                if len(waypoint_loc_tuple_from_buffer) == 1 and self._local_planner.route_end:
                    backoff = 2
                for j in range(i - backoff):
                    change = True
                    self.control_route_loc_tuple_buffer.pop(0)
                continue
            else:
                """new waypoint"""
                change = True
                self.control_route_loc_tuple_buffer.append(waypoint_loc_tuple)
                if len(self.control_route_loc_tuple_buffer) > self.look_ahaed_waypoints + 1:
                    self.control_route_loc_tuple_buffer.pop(0)
                else:
                    i += 1
        # print(waypoint_loc_tuple_from_buffer)
        # print(self.control_route_loc_tuple_buffer)
        if change:
            loc_list = []
            for loc_tuple in self.control_route_loc_tuple_buffer:
                loc_list.append(carla.Location(loc_tuple[0], loc_tuple[1], loc_tuple[2]))

            _, self.interpolated_control_route_buffer = interpolate_trajectory(CarlaDataProvider.get_world(), loc_list,
                                                                               hop_resolution=0.5)

        return change, waypoint_loc_tuple_from_buffer

    # def path_planning(self, frameId, lidar_proc_res, ego_transform, ego_speed, z_threshold):
    #
    #
    #
    #     return temp_waypoints, waypoints_deviation_dist, planner_success

    def collider_trajectory_prediction(self, object_list, actor_list, ego_trajectory_points_timestamp,
                                       ego_deviation_for_collision):
        trajectory_clear = True
        collision_detected = False
        collision_detail = None
        if ego_trajectory_points_timestamp is None or len(ego_trajectory_points_timestamp) == 0:
            return [None, collision_detected, trajectory_clear, collision_detail]
        collider_trajectory_list = []
        hero_actor = CarlaActorPool.get_hero_actor()
        ego_speed = hero_actor.get_velocity()
        ego_speed_list = [ego_speed.x, ego_speed.y, ego_speed.z]
        ego_spd_val_mps = math.hypot(ego_speed.x, ego_speed.y)
        # nearby_objects = world_object_list
        self.filtered_object_list = LidarPreprocessor.estimated_actor_and_speed_from_detected_object(object_list, hero_actor,
                                                                                                actor_list)
        #
        # print("Merged lidar visibility")
        # Utils.summarize_detected_object(self.filtered_object_list)

        for obj in self.filtered_object_list:
            # print("Predict obj {} (actor {}) trajectory".format(obj.id, obj.actor_id))
            [collider_trajectory, collision, trajectory_clear_once,
             collision_detail_once] = PathPlanner.calculate_trajectory_crossing(
                ego_trajectory_points_timestamp, ego_deviation_for_collision, obj, ego_spd_val_mps)
            collider_trajectory_list.append(collider_trajectory)
            if collision:
                collision_detected = True
            if not trajectory_clear_once:
                # break
                """not breaking for visualization"""
                trajectory_clear = False
                collision_detail = collision_detail_once
        return [collider_trajectory_list, collision_detected, trajectory_clear, collision_detail]

    def collider_trajectory_prediction_with_extrapolation(self, object_list, actor_list,
                                                          ego_trajectory_points_timestamp, ego_deviation_for_collision,
                                                          MAX_PKT_LOSS=5):

        debug = False
        """ increment pkt loss count first """
        for j in range(len(self.collider_pkt_loss_count_list)):
            self.collider_pkt_loss_count_list[j] += 1
        if debug: print(self.collider_id_list)
        if debug: print(self.collider_pkt_loss_count_list)
        if debug: print("Updating old records")

        trajectory_clear = True
        collision_detected = False
        collision_detail = None
        if ego_trajectory_points_timestamp is None or len(ego_trajectory_points_timestamp) == 0:
            return [None, collision_detected, trajectory_clear, collision_detail]
        collider_trajectory_list = []
        hero_actor = CarlaActorPool.get_hero_actor()
        ego_speed = hero_actor.get_velocity()
        ego_speed_list = [ego_speed.x, ego_speed.y, ego_speed.z]
        ego_spd_val_mps = math.hypot(ego_speed.x, ego_speed.y)
        # nearby_objects = world_object_list
        self.filtered_object_list = LidarPreprocessor.estimated_actor_and_speed_from_detected_object(object_list, hero_actor,
                                                                                                actor_list)

        # print("Merged lidar visibility")
        # Utils.summarize_detected_object(filtered_object_list)

        updated_collider_obj_list = []
        updated_collider_id_list = []
        updated_collider_pkt_loss_count_list = []

        for obj in self.filtered_object_list:
            last_obj_id = -1
            is_old_obj = False
            for i in range(len(self.collider_id_list)):
                if obj.actor_id == self.collider_id_list[i]:
                    """update old records"""
                    self.collider_pkt_loss_count_list[i] = 0
                    if debug: print("updating old records for obj id {}".format(obj.actor_id))
                    is_old_obj = True
                    break

            """add obj"""
            if not is_old_obj:
                if obj.actor_id not in updated_collider_id_list:
                    updated_collider_obj_list.append(obj)
                    updated_collider_id_list.append(obj.actor_id)
                    updated_collider_pkt_loss_count_list.append(0)

            # print("Predict obj {} (actor {}) trajectory".format(obj.id, obj.actor_id))
            [collider_trajectory, collision, trajectory_clear_once,
             collision_detail_once] = PathPlanner.calculate_trajectory_crossing(
                ego_trajectory_points_timestamp, ego_deviation_for_collision, obj, ego_spd_val_mps)
            collider_trajectory_list.append(collider_trajectory)
            if collision:
                collision_detected = True
            if not trajectory_clear_once:
                trajectory_clear = False
                collision_detail = collision_detail_once

        if debug: print(self.collider_id_list)
        if debug: print(self.collider_pkt_loss_count_list)
        if debug: print(updated_collider_id_list)
        if debug: print(updated_collider_pkt_loss_count_list)
        if debug: print("Extrapolating objs without updates")
        """ extrapolate obj without updates """
        for j in range(len(self.collider_pkt_loss_count_list)):
            if 0 < self.collider_pkt_loss_count_list[j] <= MAX_PKT_LOSS:
                obj = self.collider_objects_list[j]
                if debug: print("extrapolating obj {} without update".format(obj.actor_id))
                """update position and speed"""
                dt = 1 / Utils.CarlaFPS  # seconds
                for idx in range(3):
                    obj.esitmated_position[idx] += obj.estimated_speed[idx] * dt
                    obj.estimated_speed[idx] += obj.estimated_accel[idx] * dt
                """add old obj"""
                updated_collider_obj_list.append(obj)
                updated_collider_id_list.append(obj.actor_id)
                updated_collider_pkt_loss_count_list.append(self.collider_pkt_loss_count_list[j])

                [collider_trajectory, collision, trajectory_clear_once,
                 collision_detail_once] = PathPlanner.calculate_trajectory_crossing(
                    ego_trajectory_points_timestamp, ego_deviation_for_collision, obj, ego_spd_val_mps)
                collider_trajectory_list.append(collider_trajectory)
                if collision:
                    collision_detected = True
                if not trajectory_clear_once:
                    trajectory_clear = False
                    collision_detail = collision_detail_once

        if debug: print(self.collider_id_list)
        if debug: print(self.collider_pkt_loss_count_list)
        if debug: print(updated_collider_id_list)
        if debug: print(updated_collider_pkt_loss_count_list)

        self.collider_objects_list = updated_collider_obj_list
        self.collider_id_list = updated_collider_id_list
        self.collider_pkt_loss_count_list = updated_collider_pkt_loss_count_list

        return [collider_trajectory_list, collision_detected, trajectory_clear, collision_detail]
