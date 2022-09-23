#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
from agents.navigation.basic_agent import BasicAgent
from AVR.autocast_agents.new_agent import NewAgent
from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
import time
from AVR import Utils


class SimpleAgent(AutonomousAgent):
    """
    Simple autonomous agent to control the ego vehicle
    """

    def __init__(self, path_to_conf_file):
        super().__init__(path_to_conf_file)

        self._agent = None
        self._route_assigned = False

        self.agent_trajectory_points_timestamp = []
        self.collider_trajectory_points_timestamp = []
        self.next_target_location = None
        self.drawing_object_list = []

        self._target_speed = Utils.target_speed_kmph  # default 20 km/h
        if Utils.EvalEnv.ego_speed_kmph is not None:
            self._target_speed = Utils.EvalEnv.ego_speed_kmph
            Utils.target_speed_kmph = self._target_speed
            Utils.target_speed_mps = self._target_speed / 3.6

        self._agent_control = None

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

        sensors = []
        # BEV camera
        if Utils.TEST_INTROVIDEO:
            # for high resolution demo video
            sensors = [
                {'type': 'sensor.camera.rgb', 'x': Utils.LidarRoofForwardDistance, 'y': 0.0, 'z': Utils.LidarRange,
                 'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                 'width': 1920, 'height': 1920, 'fov': 90, 'id': 'RGB'}
            ]
        else:
            sensors = [
                {'type': 'sensor.camera.rgb', 'x': Utils.LidarRoofForwardDistance, 'y': 0.0, 'z': Utils.LidarRange,
                 'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                 'width': 720, 'height': 720, 'fov': 90, 'id': 'RGB'},
                # use same width height to align with lidar display
                ]

        # Lidar: use same width height to align with lidar display
        sensors += [{'type': 'sensor.lidar.ray_cast', 'x': Utils.LidarRoofForwardDistance, 'y': 0.0,
                     'z': Utils.LidarRoofTopDistance,  # the spawn function will add this on top of bbox.extent.z
                     'yaw': Utils.LidarYawCorrection, 'pitch': 0.0, 'roll': 0.0,
                     'range': Utils.LidarRange,
                     # set same as camera height, cuz camera fov is 90 deg, HUD can visualize in same dimension
                     'rotation_frequency': 20, 'channels': 64,
                     'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
                     'id': 'LIDAR'}
                    ]

        # Front view Stereo Camera
        sensors += [{'type': 'sensor.camera.rgb', 'x': Utils.CameraRoofForwardDistance, 'y': -0.4,
                     'z': Utils.CameraRoofTopDistance,
                     'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                     'width': Utils.CamWidth, 'height': Utils.CamHeight, 'fov': 100, 'id': 'Left'},
                    {'type': 'sensor.camera.rgb', 'x': Utils.CameraRoofForwardDistance, 'y': 0.4,
                     'z': Utils.CameraRoofTopDistance,
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

        start = time.time()
        if not self._agent:
            hero_actor = CarlaActorPool.get_hero_actor()
            if hero_actor:
                self._agent = NewAgent(hero_actor, self._target_speed)
            return control

        if not self._route_assigned:
            print("Simple agent: setting up global plan")
            if self._global_plan:
                plan = []
                for transform, road_option in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    plan.append((wp, road_option))
                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                self._route_assigned = True
                print("Global Plan set")
        else:
            control, res_vec = self._agent.run_control(input_data, timestamp)
            [self.agent_trajectory_points_timestamp, self.collider_trajectory_points_timestamp,
             self.drawing_object_list, self.next_target_location, _] = res_vec

        elapsed = time.time() - start
        if Utils.TIMEPROFILE: print("\tControl Loop: {} s".format(elapsed))
        self._agent_control = control
        return control

    def set_global_plan_from_parent(self, global_plan, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        self._global_plan_world_coord = global_plan_world_coord
        self._global_plan = global_plan
