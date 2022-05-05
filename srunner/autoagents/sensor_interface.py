#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This file containts CallBack class and SensorInterface, responsible of
handling the use of sensors for the agents
"""

import copy
import logging
import threading
import numpy as np

import carla



class CallBack(object):

    """
    Class the sensors listen to in order to receive their data each frame
    """

    def __init__(self, tag, sensor, data_provider):
        """
        Initializes the call back
        """
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor)

    def __call__(self, data):
        """
        call function
        """
        if isinstance(data, carla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        """
        parses cameras
        """
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        self._data_provider.update_sensor(tag, array, image, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        """
        parses lidar sensors
        """
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data, lidar_data.frame)

    def _parse_radar_cb(self, radar_data, tag):
        """
        parses radar sensors
        """
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        """
        parses gnss sensors
        """
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data, gnss_data.frame)

    def _parse_imu_cb(self, imu_data, tag):
        """
        parses IMU sensors
        """
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                          ], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, imu_data.frame)


class SensorInterface(object):

    """
    Class that contains all sensor data
    """

    def __init__(self):
        """
        Initializes the class
        """
        self._sensors_objects = {}
        self._data_buffers = {}
        self._data_object = {}
        self._timestamps = {}
        # self._lock = threading.Lock()

    def has_sensor(self, tag):
        if tag in self._sensors_objects:
            return True
        else:
            return False

    def register_sensor(self, tag, sensor):
        """
        Registers the sensors
        """
        print("registering sensor {}".format(tag))
        if tag in self._sensors_objects:
            print(self._sensors_objects[tag])
            raise ValueError("Duplicated sensor tag [{}]".format(tag))

        # self._lock.acquire()
        self._sensors_objects[tag] = sensor
        self._data_buffers[tag] = None
        self._data_object[tag] = None
        self._timestamps[tag] = -1
        # self._lock.release()
        print("finished registering sensor {}".format(tag))

    def destroy_sensor(self, tag):
        if tag not in self._sensors_objects:
            return
        print("destroying sensor {}".format(tag))
        # self._lock.acquire()
        if self._sensors_objects[tag] is not None:
            self._sensors_objects[tag].stop()
            self._sensors_objects[tag].destroy()
        try:
            del self._sensors_objects[tag]
            del self._data_buffers[tag]
            del self._data_object[tag]
            del self._timestamps[tag]
        except Exception as e:
            print("sensor {} already removed".format(tag))
        # self._lock.release()
        self._sensors_objects.pop(tag, None)
        self._data_buffers.pop(tag, None)
        self._data_object.pop(tag, None)
        self._timestamps.pop(tag, None)
        print("finished destroying sensor {}".format(tag))

    def update_sensor(self, tag, data, data_obj, timestamp):
        """
        Updates the sensor
        """
        if tag not in self._sensors_objects:
            raise ValueError("The sensor with tag [{}] has not been created!".format(tag))
        self._data_buffers[tag] = data
        self._data_object[tag] = data_obj
        self._timestamps[tag] = timestamp
        # print("Sensor Updated")

    def all_sensors_ready(self):
        """
        Checks if all the sensors have sent data at least once
        """
        for key in self._sensors_objects:
            if self._data_buffers[key] is None:
                return False
        return True

    def get_data(self):
        """
        Returns the data of a sensor
        """
        # self._lock.acquire()
        data_dict = {}
        for key in self._sensors_objects:
            data_dict[key] = (self._timestamps[key], self._data_buffers[key])
        # self._lock.release()
        return data_dict

    def get_data_obj(self):
        # print("getting data object")
        # self._lock.acquire()
        data_obj_dict = {}
        # print(self._sensors_objects.keys())
        # print(self._timestamps.keys())
        # print(self._data_object.keys())
        for key in self._sensors_objects.keys():
            data_obj_dict[key] = (self._timestamps[key], self._data_object[key])
        # self._lock.release()
        # print("finished getting data object")
        return data_obj_dict

    def get_data_by_id(self, sensor_id):
        """
        Returns the data of a sensor
        """
        if sensor_id in self._sensors_objects:
            return self._data_buffers[sensor_id]
        else:
            return None

    def get_data_obj_by_id(self, sensor_id):
        if sensor_id in self._sensors_objects:
            return self._data_object[sensor_id]
        else:
            return None
