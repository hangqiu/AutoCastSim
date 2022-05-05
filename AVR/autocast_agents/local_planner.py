#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

from enum import Enum
from collections import deque
import random

import carla
from agents.navigation.controller import VehiclePIDController
from agents.tools.misc import distance_vehicle, draw_waypoints
import math
import numpy as np
from AVR import Utils


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control
    and the other for the longitudinal control (cruise speed).

    When multiple paths are available (intersections) this local planner makes a random choice.
    """

    # minimum distance to target waypoint as a percentage (e.g. within 90% of
    # total distance)
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict=None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
            dt -- time difference between physics control in seconds. This is typically fixed from server side
                  using the arguments -benchmark -fps=F . In this case dt = 1/F

            target_speed -- desired cruise speed in Km/h

            sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

            lateral_control_dict -- dictionary of arguments to setup the lateral PID controller
                                    {'K_P':, 'K_D':, 'K_I':, 'dt'}

            longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                        {'K_P':, 'K_D':, 'K_I':, 'dt'}
        """
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_pid_controller = None
        self._steering_pid_controller = None
        self._global_plan = None
        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen=20000)
        self._buffer_size = 5
        self._waypoint_buffer = deque(maxlen=self._buffer_size)

        self.past_steering = None
        self.route_end = False

        # initializing controller
        self._init_controller(opt_dict)

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20 # default was 20 Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        # self._min_distance = 10 # autocast: use constant, not adapting to speed
        self._max_brake = 0.75
        self._max_throt = 0.75
        self._max_steer = 1.0
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.2,
            'K_I': 0.07,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 0.05,
            'dt': self._dt}

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius = self._target_speed * \
                                        opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']
            if 'max_throttle' in opt_dict:
                self._max_throt = opt_dict['max_throttle']
            if 'max_brake' in opt_dict:
                self._max_brake = opt_dict['max_brake']
            if 'max_steering' in opt_dict:
                self._max_steer = opt_dict['max_steering']

        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_pid_controller = VehiclePIDController(self._vehicle,
                                                            args_lateral=args_lateral_dict,
                                                            args_longitudinal=args_longitudinal_dict,
                                                            max_throttle=self._max_throt,
                                                            max_brake=self._max_brake,
                                                            max_steering=self._max_steer)

        self._global_plan = False

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))

        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k=200)
        """Avr"""
        self._steering_pid_controller = PIDLateralController_Custom(self._vehicle, **args_lateral_dict)
        self.past_steering = self._vehicle.get_control().steer
        print("Finished Initialization Controller")

    def set_speed(self, speed):
        """
        Request new target speed.

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = _retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))

    def set_global_plan(self, current_plan):
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem)
        self._target_road_option = RoadOption.LANEFOLLOW
        self._global_plan = True

    def run_step(self, goal=None):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon? => add more!
        if not self._global_plan and len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k=100)

        if len(self._waypoints_queue) == 0:
            self.route_end = True

        #   Buffering the waypoints
        if not self._waypoint_buffer:
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoint_buffer.append(
                        self._waypoints_queue.popleft())
                else:
                    break

        self.evict_waypoint_buffer()

        # current vehicle waypoint
        # self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        control = Utils.default_control()
        if len(self._waypoint_buffer) == 0:
            return control
        # target waypoint
        if goal is None:
            self.target_waypoint, self._target_road_option = self._waypoint_buffer[-1]
            # # move using PID controllers
            # control = self._vehicle_pid_controller.run_step(self._target_speed, self.target_waypoint)
            return control
        elif goal is 'Forward':
            print("Going forward")
            control = self.controller_run_step(self._target_speed, None)
        else:
            """ use default pid for long control, customize pid for lat control, cuz no constructor for waypoint """
            target_waypoint_transform = carla.Transform(carla.Location(x=goal[0], y=goal[1], z=goal[2]))
            # control.throttle = self._vehicle_pid_controller._lon_controller.run_step(self._target_speed)
            # control.steer = self._steering_pid_controller.run_step(target_waypoint_transform)
            control = self.controller_run_step(self._target_speed, target_waypoint_transform)


        # if debug:
        #     draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], self._vehicle.get_location().z + 1.0)

        return control

    def controller_run_step(self, target_speed, target_waypoint_transform):
        """
                Execute one step of control invoking both lateral and longitudinal
                PID controllers to reach a target waypoint
                at a given target_speed.

                    :param target_speed: desired vehicle speed
                    :param target_waypoint_transform: target location encoded as a transform
                    :return: distance (in meters) to the waypoint
                """

        acceleration = self._vehicle_pid_controller._lon_controller.run_step(target_speed)
        if target_waypoint_transform:
            current_steering = self._steering_pid_controller.run_step(target_waypoint_transform)
        else:
            current_steering = 0.0

        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self._max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self._max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        # if current_steering > self.past_steering + 0.1:
        #     current_steering = self.past_steering + 0.1
        # elif current_steering < self.past_steering - 0.1:
        #     current_steering = self.past_steering - 0.1

        # if current_steering >= 0:
        #     steering = min(self._max_steer, current_steering)
        # else:
        #     steering = max(-self._max_steer, current_steering)
        steering = current_steering
        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control

    def evict_waypoint_buffer(self):
        # purge the queue of obsolete waypoints
        vehicle_transform = self._vehicle.get_transform()
        max_index = -1

        # print("Local Planner: eviction distance {}".format(self._min_distance))
        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            dist = distance_vehicle(waypoint, vehicle_transform)
            # print(dist)
            if dist < self._min_distance:
                max_index = i
        # print(max_index)
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoint_buffer.popleft()


class PIDLateralController_Custom:
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def run_step(self, target_waypoint_transform):
        """
        Execute one step of lateral control to steer the vehicle towards a certain waypoin.

        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represent maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(target_waypoint_transform, self._vehicle.get_transform())

    def _pid_control(self, target_waypoint_transform, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y=math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([target_waypoint_transform.location.x -
                          v_begin.x, target_waypoint_transform.location.y -
                          v_begin.y, 0.0])
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        # 0,9,8
        # return np.clip((self._K_P * _dot) + (self._K_D * _de /
        #                                      self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)
        # 0.9.9
        return np.clip((self._K_P * _dot) + (self._K_D * _de) + (self._K_I * _ie), -1.0, 1.0)

def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
