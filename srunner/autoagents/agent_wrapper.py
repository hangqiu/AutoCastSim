#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Wrapper for autonomous agents required for tracking and checking of used sensors
"""

from __future__ import print_function

import copy
import threading

import carla

from srunner.autoagents.sensor_interface import CallBack
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from AVR.BeaconList import BeaconList
from AVR.Collaborator import Collaborator
from AVR import Utils, Comm


class AgentWrapper(object):

    """
    Wrapper for autonomous agents required for tracking and checking of used sensors
    """

    _agent = None
    _sensors_list = []
    _collaborator_dict = dict()
    _beacon_list = None
    _collaborator_id_for_hud = 0

    _schedule = None
    _schedule_frame = -1
    _schedule_lock = threading.Lock()

    def __init__(self, agent):
        """
        Set the autonomous agent
        """
        self._agent = agent

    def __call__(self):
        """
        Pass the call directly to the agent
        """
        return self._agent()

    def update_schedule(self, frameId, schedule_vector, beaconlist):
        ret = []
        self._schedule_lock.acquire()
        if frameId > self._schedule_frame:
            self._schedule=schedule_vector
            self._schedule_frame = frameId
        ret = copy.deepcopy(self._schedule)
        self._schedule_lock.release()
        return ret

    def setup_sensors(self, vehicle, sensors=None, debug_mode=False):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        bp_library = CarlaDataProvider.get_world().get_blueprint_library()
        if sensors is None:
            sensors = self._agent.sensors()
        for sensor_spec in sensors:
            # These are the sensors spawned on the carla world
            bp = bp_library.find(str(sensor_spec['type']))
            if sensor_spec['type'].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(sensor_spec['width']))
                bp.set_attribute('image_size_y', str(sensor_spec['height']))
                bp.set_attribute('fov', str(sensor_spec['fov']))
                z_extra = 0
                if sensor_spec['id'] != 'RGB':
                    z_extra = vehicle.bounding_box.extent.z * 2
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                 z=sensor_spec['z'] + z_extra)
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                 roll=sensor_spec['roll'],
                                                 yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.lidar'):
                bp.set_attribute('range', str(sensor_spec['range']))
                bp.set_attribute('rotation_frequency', str(sensor_spec['rotation_frequency']))
                bp.set_attribute('channels', str(sensor_spec['channels']))
                bp.set_attribute('upper_fov', str(sensor_spec['upper_fov']))
                bp.set_attribute('lower_fov', str(sensor_spec['lower_fov']))
                bp.set_attribute('points_per_second', str(sensor_spec['points_per_second']))
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                 z=sensor_spec['z'] + vehicle.bounding_box.extent.z * 2)
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                 roll=sensor_spec['roll'],
                                                 yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.other.gnss'):
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                 z=sensor_spec['z'])
                sensor_rotation = carla.Rotation()

            # create sensor
            sensor_transform = carla.Transform(sensor_location, sensor_rotation)
            sensors = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, vehicle)
            # setup callback
            sensors.listen(CallBack(str(vehicle.id) + "_" + str(sensor_spec['id']), sensors, self._agent.sensor_interface))
            self._sensors_list.append(sensors)

        while not self._agent.all_sensors_ready():
            if debug_mode:
                print(" waiting for one data reading from sensors...")
            CarlaDataProvider.get_world().tick()

    def destroy_sensors(self, vehicle_id, sensors=None, debug_mode=False):
        if sensors is None:
            sensors = self._agent.sensors()
        for sensor_spec in sensors:
            tag = str(vehicle_id) + "_" + str(sensor_spec['id'])
            self._agent.sensor_interface.destroy_sensor(tag)


    # def try_setup_lidar_on_other_vehicle(self, actor, debug_mode=False):
    #     # print("Setting up sensor for actor {}".format(actor.id))
    #     sensor_id = str(actor.id) + "_LIDAR"
    #     if self._agent.sensor_interface.has_sensor(sensor_id):
    #         return
    #     if debug_mode:
    #         print("actor {} ({}) comes in vincinity, spawning sensor".format(actor.id, actor.type_id))
    #     sensor = [{'type': 'sensor.lidar.ray_cast', 'x': Utils.LidarRoofForwardDistance, 'y': 0.0,
    #                'z': Utils.LidarRoofTopDistance, # 30 cm above vehicle bbox
    #                'yaw': 0.0, 'pitch': 0.0,
    #                'roll': 0.0,
    #                'range': 50, # set same as camera height, cuz camera fov is 90 deg, HUD can visualize in same dimension
    #                'rotation_frequency': 20, 'channels': 64,
    #                'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
    #                'id': 'LIDAR'}]
    #     try:
    #         self.setup_sensors(actor, sensor)
    #     except Exception as e:
    #         print("Failed to spawn lidar on actor id {}".format(actor.id))
    #         pass

    def setup_collaborator(self, vehicle, sharing=False):
        """
        setup collaborator on all actors, in charge of handling communication and sensor processing
        """
        self._collaborator_dict[vehicle.id] = Collaborator(vehicle, self._agent, self, sharing)

    def destroy_collaborator(self, vehicle_id):
        if vehicle_id in self._collaborator_dict:
            self._collaborator_dict[vehicle_id].destroy()
            del self._collaborator_dict[vehicle_id]

    def tick_collaborators(self):
        destroy_list = []
        for id in self._collaborator_dict.keys():
            c = self._collaborator_dict[id]
            if c:
                if c.is_alive():
                    c.tick()
                else:
                    self.destroy_sensors(id)
                    c.destroy()
                    destroy_list.append(id)
        for id in destroy_list:
            del self._collaborator_dict[id]

    def join_collaborators(self):
        for c in self._collaborator_dict.values():
            if c:
                c.tick_join()

    def get_collaborator_for_hud(self, vehicle_id):
        if vehicle_id in self._collaborator_dict:
            return self._collaborator_dict[vehicle_id]
        else:
            return None

    def setup_beacon(self):
        """
        setup beacon, in charge of controll channel setting and consensus management (incorporate into RSU in the future)
        """
        self._beacon_list = BeaconList()

    def get_beacon_list(self):
        return self._beacon_list.get_beacon_list()

    def append_beacon_list(self, beacon):
        if self._beacon_list is not None:
            self._beacon_list.append_beacon_list(beacon)

    def tick_beacon(self):
        self._beacon_list.tick()

    def has_sensor(self, tag):
        return self._agent.sensor_interface.has_sensor(tag)

    def get_sensor(self, tag):
        return self._agent.sensor_interface.get_data_by_id(tag)

    def get_sensor_obj(self, tag):
        return self._agent.sensor_interface.get_data_obj_by_id(tag)

    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].stop()
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        self._sensors_list = []

        destoryer_list= []
        for c in self._collaborator_dict.values():
            if c is not None:
                d = threading.Thread(target=c.destroy)
                d.start()
                destoryer_list.append(d)
        for d in destoryer_list:
            d.join()
        self._collaborator_dict = dict()

        if self._beacon_list is not None:
            del self._beacon_list
            self._beacon_list = None
