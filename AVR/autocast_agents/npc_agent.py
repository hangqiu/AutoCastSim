#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
from agents.navigation.basic_agent import BasicAgent

from AVR import Utils
from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class NpcAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False
    agent_trajectory_points_timestamp = []
    collider_trajectory_points_timestamp = []
    next_target_location = None
    drawing_object_list = []
    _agent_control = None

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self._route_assigned = False
        self._agent = None

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}


        """

        # sensors = [
        #     # {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 50.0, 'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
        #     #  'width': 1280, 'height': 720, 'fov': 100, 'id': 'RGB'},
        #     # {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.6, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        #     #  'width': 1280, 'height': 720, 'fov': 100, 'id': 'RGB'},
        #     {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 50.0, 'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
        #      'width': 720, 'height': 720, 'fov': 90, 'id': 'RGB'},  # use same width height to align with lidar display
        #     {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
        #      'range': 100, 'rotation_frequency': 20, 'channels': 64,
        #      'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
        #      'id': 'LIDAR'},
        # ]
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': Utils.LidarRoofForwardDistance, 'y': 0.0, 'z': Utils.LidarRange, 'roll': 0.0,
             'pitch': -90.0, 'yaw': 0.0,
             'width': 720, 'height': 720, 'fov': 90, 'id': 'RGB'},  # use same width height to align with lidar display
            {'type': 'sensor.lidar.ray_cast', 'x': Utils.LidarRoofForwardDistance, 'y': 0.0,
             'z': Utils.LidarRoofTopDistance,  # the spawn function will add this on top of bbox.extent.z
             'yaw': Utils.LidarYawCorrection, 'pitch': 0.0, 'roll': 0.0,
             'range': Utils.LidarRange,
             # set same as camera height, cuz camera fov is 90 deg, HUD can visualize in same dimension
             'rotation_frequency': 20, 'channels': 64,
             'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
             'id': 'LIDAR'},

            {'type': 'sensor.camera.rgb', 'x': Utils.LidarRoofForwardDistance, 'y': -0.4, 'z': Utils.LidarRoofTopDistance,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': Utils.CamWidth, 'height': Utils.CamHeight, 'fov': 100, 'id': 'Left'},
            {'type': 'sensor.camera.rgb', 'x': Utils.LidarRoofForwardDistance, 'y': 0.4, 'z': Utils.LidarRoofTopDistance,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': Utils.CamWidth, 'height': Utils.CamHeight, 'fov': 100, 'id': 'Right'},
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        if not self._agent:
            hero_actor = None
            for actor in CarlaDataProvider.get_world().get_actors():
                if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                    hero_actor = actor
                    break
            if hero_actor:
                self._agent = BasicAgent(hero_actor)

            return control

        if not self._route_assigned:
            if self._global_plan:
                plan = []

                for transform, road_option in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    plan.append((wp, road_option))
                    print("Global Plan: trans {}, waypoint {}".format(transform.location, wp.transform.location))

                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                self._route_assigned = True

        else:
            control = self._agent.run_step()
            print("Waypoint queue length {}, buffer len {}".format(len(self._agent._local_planner._waypoints_queue),
                                                                   len(self._agent._local_planner._waypoint_buffer)))

        self._agent_control = control
        return control
