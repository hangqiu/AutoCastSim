#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

import time
from threading import Thread
import cv2
import numpy as np
from AVR import Utils

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import carla

from srunner.autoagents.autonomous_agent import AutonomousAgent


class HumanAgent(AutonomousAgent):

    """
    Human agent to control the ego vehicle via keyboard
    """

    current_control = Utils.default_control()

    def __init__(self, path_to_conf_file):
        super().__init__(path_to_conf_file)

        self._agent = None
        self._route_assigned = False

        self.agent_trajectory_points_timestamp = []
        self.collider_trajectory_points_timestamp = []
        self.next_target_location = None
        self.drawing_object_list = []

        self._target_speed = Utils.target_speed_kmph #  default 20 km/h
        if Utils.EvalEnv.ego_speed_kmph is not None:
            self._target_speed = Utils.EvalEnv.ego_speed_kmph
            Utils.target_speed_kmph = self._target_speed
            Utils.target_speed_mps = self._target_speed / 3.6

        self._agent_control = None

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        Utils.HUMAN_AGENT = True


    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            ['sensor.camera.rgb', {'x':x_rel, 'y': y_rel, 'z': z_rel,
                                   'yaw': yaw, 'pitch': pitch, 'roll': roll,
                                   'width': width, 'height': height, 'fov': fov}, 'Sensor01'],
            ['sensor.camera.rgb', {'x':x_rel, 'y': y_rel, 'z': z_rel,
                                   'yaw': yaw, 'pitch': pitch, 'roll': roll,
                                   'width': width, 'height': height, 'fov': fov}, 'Sensor02'],

            ['sensor.lidar.ray_cast', {'x':x_rel, 'y': y_rel, 'z': z_rel,
                                       'yaw': yaw, 'pitch': pitch, 'roll': roll}, 'Sensor03']
        ]

        """
        sensors = [
            # {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #  'width': 1280, 'height': 720, 'fov': 100, 'id': 'RGB'},
            # {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
            #  'range': 100, 'rotation_frequency': 20, 'channels': 64, 'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
            #  'id': 'LIDAR'},
            {'type': 'sensor.camera.rgb', 'x': Utils.LidarRoofForwardDistance, 'y': 0.0, 'z': Utils.LidarRange,
             'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
             'width': 720, 'height': 720, 'fov': 90, 'id': 'RGB'},  # use same width height to align with lidar display
            {'type': 'sensor.lidar.ray_cast', 'x': Utils.LidarRoofForwardDistance, 'y': 0.0,
             'z': Utils.LidarRoofTopDistance,  # the spawn function will add this on top of bbox.extent.z
             'yaw': Utils.LidarYawCorrection, 'pitch': 0.0, 'roll': 0.0,
             'range': Utils.LidarRange,
             # set same as camera height, cuz camera fov is 90 deg, HUD can visualize in same dimension
             'rotation_frequency': 20, 'channels': 64,
             'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
             'id': 'LIDAR'},

            {'type': 'sensor.camera.rgb', 'x': Utils.LidarRoofForwardDistance, 'y': -0.4,
             'z': Utils.LidarRoofTopDistance,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': Utils.CamWidth, 'height': Utils.CamHeight, 'fov': 100, 'id': 'Left'},
            {'type': 'sensor.camera.rgb', 'x': Utils.LidarRoofForwardDistance, 'y': 0.4,
             'z': Utils.LidarRoofTopDistance,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': Utils.CamWidth, 'height': Utils.CamHeight, 'fov': 100, 'id': 'Right'},
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        # time.sleep(0.1)
        return self.current_control

    def destroy(self):
        """
        Cleanup
        """
