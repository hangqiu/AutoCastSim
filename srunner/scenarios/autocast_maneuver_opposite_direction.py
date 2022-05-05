#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Vehicle Maneuvering In Opposite Direction:

Vehicle is passing another vehicle in a rural area, in daylight, under clear
weather conditions, at a non-junction and encroaches into another
vehicle traveling in the opposite direction.
"""

from six.moves.queue import Queue   # pylint: disable=relative-import

import math
import py_trees
import carla

from AVR import Utils
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      ActorSource,
                                                                      ActorSink,
                                                                      WaypointFollower,
                                                                      KeepVelocity,
                                                                      SetInitSpeed)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, InTriggerRegion_GlobalUtilFlag
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_waypoint_in_distance


class AutoCastManeuverOppositeDirection(BasicScenario):

    """
    "Vehicle Maneuvering In Opposite Direction" (Traffic Scenario 06)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 obstacle_type='vehicle', timeout=120):
        """
        Setup all relevant parameters and create scenario
        obstacle_type -> flag to select type of leading obstacle. Values: vehicle, barrier
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()

        self._first_vehicle_location = 6 # was 50, set to 30 for eval, 10 for faster testing
        if Utils.EvalEnv.ego_distance is not None:
            self._first_vehicle_location = Utils.EvalEnv.ego_distance

        self._collider_trigger_distance = self._first_vehicle_location + 40
        if Utils.EvalEnv.collider_distance is not None:
            self._collider_trigger_distance = self._first_vehicle_location + Utils.EvalEnv.collider_distance

        self._second_vehicle_location = self._collider_trigger_distance
        if Utils.EvalEnv.collider_accel_distance is not None:
            self._second_vehicle_location = self._collider_trigger_distance + Utils.EvalEnv.collider_accel_distance

        self._ego_vehicle_drive_distance = self._second_vehicle_location * 2
        self._start_distance = self._first_vehicle_location * 1

        self._opposite_speed = 30 / 3.6  # was 5.56 m/s for 20km/h
        if Utils.EvalEnv.collider_speed_kmph is not None:
            self._opposite_speed = Utils.EvalEnv.collider_speed_kmph / 3.6

        self._occluder_speed = 10 / 3.6
        self._source_gap = 40   # m was 40 m
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._source_transform = None
        self._sink_location = None
        self._blackboard_queue_name = 'ManeuverOppositeDirection/actor_flow_queue'
        self._queue = py_trees.blackboard.Blackboard().set(self._blackboard_queue_name, Queue())
        self._obstacle_type = obstacle_type
        self._first_actor_transform = None
        self._second_actor_transform = None
        self._third_actor_transform = None

        self._occluder = None
        self._collider = None
        self._second_occluder = None

        # Timeout of scenario in seconds
        # self.timeout = timeout
        self.timeout = 40.0 #Autocast timeout

        super(AutoCastManeuverOppositeDirection, self).__init__(
            "ManeuverOppositeDirection",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        first_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        second_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_vehicle_location)
        second_actor_waypoint = second_actor_waypoint.get_left_lane()

        first_actor_transform = carla.Transform(
            first_actor_waypoint.transform.location,
            first_actor_waypoint.transform.rotation)

        # for more occlusion case
        first_actor_transform.location.y += 0.8

        if self._obstacle_type == 'vehicle':
            first_actor_model = 'vehicle.carlamotors.carlacola'
        else:
            first_actor_transform.rotation.yaw += 90
            first_actor_model = 'static.prop.streetbarrier'
            second_prop_waypoint = first_actor_waypoint.next(2.0)[0]
            position_yaw = second_prop_waypoint.transform.rotation.yaw + 90
            offset_location = carla.Location(
                0.50 * second_prop_waypoint.lane_width * math.cos(math.radians(position_yaw)),
                0.50 * second_prop_waypoint.lane_width * math.sin(math.radians(position_yaw)))
            second_prop_transform = carla.Transform(
                second_prop_waypoint.transform.location + offset_location, first_actor_transform.rotation)
            second_prop_actor = CarlaActorPool.request_new_actor(first_actor_model, second_prop_transform)
            second_prop_actor.set_simulate_physics(True)
            self._third_actor_transform = second_prop_transform

        first_actor = CarlaActorPool.request_new_actor(first_actor_model, first_actor_transform)
        first_actor.set_simulate_physics(True)
        self._occluder = first_actor
        self.other_actors.append(first_actor)
        self._first_actor_transform = first_actor_transform

        if not Utils.NO_COLLIDER:
            rolename = 'scenario'
            if Utils.PASSIVE_COLLIDER:
                rolename = Utils.PASSIVE_ACTOR_ROLENAME
            second_actor = CarlaActorPool.request_new_actor('vehicle.audi.tt', second_actor_waypoint.transform, rolename=rolename)
            self.other_actors.append(second_actor)
            self._collider = second_actor
            self._second_actor_transform = second_actor_waypoint.transform
            if self._obstacle_type != 'vehicle':
                self.other_actors.append(second_prop_actor)
                self._second_occluder = second_prop_actor

        self._source_transform = second_actor_waypoint.transform
        # sink_waypoint = second_actor_waypoint.next(1)[0]
        # while not sink_waypoint.is_intersection:
        #     sink_waypoint = sink_waypoint.next(1)[0]
        """ sink waypoint is too far, red light will cuz the backlog """
        """ move sink point closer to evict cars """
        sink_waypoint = self._reference_waypoint.get_left_lane()
        sink_waypoint = sink_waypoint.next(45)[0]
        self._sink_location = sink_waypoint.transform.location
        # print("Sink location: {}".format(self._sink_location))

        sink_behind_ego_waypoint = self._reference_waypoint.previous(50)[0]
        self._sink_behind_ego_location = sink_behind_ego_waypoint.transform.location


    def _create_behavior(self):
        """
        The behavior tree returned by this method is as follows:
        The ego vehicle is trying to pass a leading vehicle in the same lane
        by moving onto the oncoming lane while another vehicle is moving in the
        opposite direction in the oncoming lane.
        """

        print("Creating scenario 6 behavior")
        # Leaf nodes

        ego_drive_distance = DriveDistance(self.ego_vehicles[0], self._ego_vehicle_drive_distance)


        # keep_velocity_occluder = KeepVelocity(
        #     self.other_actors[0],
        #     self._occluder_speed)

        # collider_trigger_waypoint_right, _ = get_waypoint_in_distance(self._reference_waypoint, self._collider_trigger_distance)
        # collider_trigger_waypoint = collider_trigger_waypoint_right.get_left_lane()
        # collider_trigger_location = collider_trigger_waypoint.transform.location
        # collider_distance_trigger = InTriggerRegion_GlobalUtilFlag(
        #     self.other_actors[1],
        #     collider_trigger_location.x - 2, collider_trigger_location.x + 2,
        #     collider_trigger_location.y - 2, collider_trigger_location.y + 2)

        # Non-leaf nodes
        parallel_root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # Building tree

        if not Utils.NO_COLLIDER:
            actor_source = ActorSource(
                ['vehicle.audi.tt', 'vehicle.tesla.model3', 'vehicle.nissan.micra'],
                self._source_transform, self._source_gap, self._blackboard_queue_name)

            waypoint_follower = WaypointFollower(
                self._collider, self._opposite_speed,
                blackboard_queue_name=self._blackboard_queue_name, avoid_collision=True)
            parallel_root.add_child(actor_source)
            parallel_root.add_child(waypoint_follower)

        # parallel_root.add_child(keep_velocity_occluder)
        # parallel_root.add_child(collider_distance_trigger)

        actor_sink = ActorSink(self._sink_location, 3)
        parallel_root.add_child(actor_sink)

        behind_ego_actor_sink = ActorSink(self._sink_behind_ego_location, 3)
        parallel_root.add_child(behind_ego_actor_sink)
        parallel_root.add_child(ego_drive_distance)

        scenario_sequence = py_trees.composites.Sequence("ManeuverOppositeDirection")
        scenario_sequence.add_child(ActorTransformSetter(self._occluder, self._first_actor_transform))


        if not Utils.NO_COLLIDER:
            scenario_sequence.add_child(ActorTransformSetter(self._collider, self._second_actor_transform))
            if self._third_actor_transform is not None:
                scenario_sequence.add_child(ActorTransformSetter(self._second_occluder, self._third_actor_transform))
            init_spd = SetInitSpeed(self._collider, self._opposite_speed)
            scenario_sequence.add_child(init_spd)

        parallel_seq = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel_seq.add_child(ego_drive_distance)
        parallel_seq.add_child(actor_sink)
        parallel_seq.add_child(behind_ego_actor_sink)

        scenario_sequence.add_child(parallel_root)
        scenario_sequence.add_child(parallel_seq)

        scenario_sequence.add_child(ActorDestroy(self._occluder))
        if not Utils.NO_COLLIDER:
            scenario_sequence.add_child(ActorDestroy(self._collider))
            if self._third_actor_transform is not None:
                scenario_sequence.add_child(ActorDestroy(self._second_occluder))

        print("Finished scenario 6 behavior")

        return scenario_sequence

    def _setup_scenario_trigger(self, config):
        # Utils.InTriggerRegion_GlobalUtilFlag = False
        return None

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
