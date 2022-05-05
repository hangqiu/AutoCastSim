#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Collection of traffic scenarios where the ego vehicle (hero)
is making a left turn
"""

from six.moves.queue import Queue  # pylint: disable=relative-import

import py_trees
import carla
from agents.navigation.local_planner import RoadOption

from AVR import Utils
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      ActorSource,
                                                                      ActorSink,
                                                                      WaypointFollower, TrafficLightManipulator)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerRegion, DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import generate_target_waypoint_list


class AutoCastIntersectionUnprotectedLeftTurn(BasicScenario):

    """
    Implementation class for Hero
    Vehicle turning left at signalized junction scenario,
    Traffic Scenario 08.

    This is a single ego vehicle scenario
    """


    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=30):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._brake_value = 0.5
        # self._ego_distance = 110
        self._ego_distance = 40
        # self._traffic_light = None
        self._other_actor_transform = None
        self._blackboard_queue_name = 'AutoCastIntersectionUnprotectedLeftTurn/actor_flow_queue'
        self._queue = py_trees.blackboard.Blackboard().set(self._blackboard_queue_name, Queue())
        # self._initialized = True
        self._source_gap = 20
        self._collider_accel_dist = 1 # was 15
        if Utils.EvalEnv.collider_accel_distance is not None:
            self._collider_accel_dist = Utils.EvalEnv.collider_accel_distance
            print("COLLIDER ACCEL DIST:", self._collider_accel_dist)
        self._target_vel = 40 / 3.6 # 20km/h
        if Utils.EvalEnv.collider_speed_kmph is not None:
            self._target_vel = Utils.EvalEnv.collider_speed_kmph / 3.6

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(ego_vehicles[0], False)
        # "Topology" of the intersection
        self._annotations = CarlaDataProvider.annotate_trafficlight_in_group(self._traffic_light)
        # TODO: use atomic behavior instead later for multiple scenarios along one route.
        #  For now, assuming only one scenario per route, and starting position is right before trigger location
        RED = carla.TrafficLightState.Red
        YELLOW = carla.TrafficLightState.Yellow
        GREEN = carla.TrafficLightState.Green
        CarlaDataProvider.update_light_states(
            self._traffic_light,
            self._annotations,
            {'ego': GREEN, 'ref': GREEN, 'left': RED, 'right': RED, 'opposite': GREEN},
            freeze=True)

        self._occluder = None
        self._collider = None

        # self.timeout = timeout  # Timeout of scenario in seconds
        self.timeout = 30.0  # Autocast timeout

        super(AutoCastIntersectionUnprotectedLeftTurn, self).__init__("AutoCastIntersectionUnprotectedLeftTurn",
                                                         ego_vehicles,
                                                         config,
                                                         world,
                                                         debug_mode,
                                                         criteria_enable=criteria_enable)



        # self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)
        # traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        # if self._traffic_light is None or traffic_light_other is None:
        #     raise RuntimeError("No traffic light for the given location found")
        # self._traffic_light.set_state(carla.TrafficLightState.Green)
        # self._traffic_light.set_green_time(self.timeout)
        # # other vehicle's traffic light
        # traffic_light_other.set_state(carla.TrafficLightState.Green)
        # traffic_light_other.set_green_time(self.timeout)

    def _initialize_actors(self, config):
        """
        setup the other actor (collider) in the right next lane or the transform in the config
        setup a blocking view truck in the config transform lane at intersection
        """
        other_actor_waypoint = CarlaDataProvider.get_map().get_waypoint(
            config.other_actors[0].transform.location)

        # self._other_actor_transform = other_actor_waypoint.get_right_lane().transform
        ################TODO
        #if self._collider_accel_dist > 0:
        #    self._other_actor_transform = other_actor_waypoint.get_right_lane().previous(self._collider_accel_dist)[0].transform
        #else:
        #    self._other_actor_transform = other_actor_waypoint.get_right_lane().transform
        ################TODO
        if self._collider_accel_dist > 0:
            self._other_actor_transform = other_actor_waypoint.previous(self._collider_accel_dist)[0].transform
        else:
            self._other_actor_transform = other_actor_waypoint.transform

        # first_vehicle_transform = carla.Transform(
        #     carla.Location(self._other_actor_transform.location.x,
        #                    self._other_actor_transform.location.y,
        #                    self._other_actor_transform.location.z - 500),
        #     self._other_actor_transform.rotation)

        if not Utils.NO_COLLIDER:
            rolename = 'scenario'
            if Utils.PASSIVE_COLLIDER:
                rolename = Utils.PASSIVE_ACTOR_ROLENAME
            first_vehicle = CarlaActorPool.request_new_actor('vehicle.audi.tt', self._other_actor_transform, rolename=rolename)
            # first_vehicle.set_transform(first_vehicle_transform)
            # first_vehicle.set_simulate_physics(enabled=False)
            self._collider = first_vehicle
            self.other_actors.append(first_vehicle)

        # block view left-turning vehicle in the opposite direction
        # while other_actor_waypoint:
        #     print("finding intersection")
        #     if other_actor_waypoint.is_junction:
        #         print("junction found")
        #         break
        #     if other_actor_waypoint.is_intersection:
        #         print("intersection found")
        #         break
        other_actor_waypoint_intersection = other_actor_waypoint.next_until_lane_end(1)[-1]
        other_actor_waypoint_intersection = other_actor_waypoint_intersection.next(5)[0]

        self._blocking_truck_transform = carla.Transform(
            carla.Location(other_actor_waypoint_intersection.transform.location.x,
                           other_actor_waypoint_intersection.transform.location.y-3,
                           1),
            carla.Rotation(other_actor_waypoint_intersection.transform.rotation.pitch,
                           other_actor_waypoint_intersection.transform.rotation.yaw-30, # was +30 for left turning compensation
                           other_actor_waypoint_intersection.transform.rotation.roll)
            )
        self._occluder = CarlaActorPool.request_new_actor('vehicle.carlamotors.carlacola',
                                                          self._blocking_truck_transform)
        self._occluder.set_simulate_physics(True)
        self.other_actors.append(self._occluder)

    def _create_behavior(self):
        """
        Hero vehicle is turning left in an urban area,
        at a signalized intersection, while other actor coming straight
        .The hero actor may turn left either before other actor
        passes intersection or later, without any collision.
        After 80 seconds, a timeout stops the scenario.
        """

        # traffic_hack = TrafficLightManipulator(self.ego_vehicles[0], subtype='S8left')


        sequence = py_trees.composites.Sequence("AutoCastIntersectionUnprotectedLeftTurn")

        # Selecting straight path at intersection
        # target_waypoint = generate_target_waypoint(
        #     CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), 0)
        # # Generating waypoint list till next intersection
        # plan = []
        # wp_choice = target_waypoint.next(1.0)
        # while not wp_choice[0].is_intersection:
        #     target_waypoint = wp_choice[0]
        #     plan.append((target_waypoint, RoadOption.LANEFOLLOW))
        #     wp_choice = target_waypoint.next(1.0)
        # adding flow of actors
        actor_source = ActorSource(
            ['vehicle.tesla.model3', 'vehicle.audi.tt'],
            self._other_actor_transform, self._source_gap, self._blackboard_queue_name)
        # destroying flow of actors
        # actor_sink = ActorSink(plan[-1][0].transform.location, 10)
        # follow waypoints untill next intersection
        # move_actor = WaypointFollower(self.other_actors[0], self._target_vel, plan=plan,
        #                               blackboard_queue_name=self._blackboard_queue_name, avoid_collision=True)

        # Generate plan for WaypointFollower
        turn = 0  # drive straight ahead
        plan = []

        # generating waypoints until intersection (target_waypoint)
        _, target_waypoint = generate_target_waypoint_list(
            CarlaDataProvider.get_map().get_waypoint(self._other_actor_transform.location), turn)

        # Generating waypoint list till next intersection
        plan.append((target_waypoint, RoadOption.LANEFOLLOW))
        wp_choice = target_waypoint.next(5.0)
        while len(wp_choice) == 1:
            target_waypoint = wp_choice[0]
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(5.0)

        move_actor = WaypointFollower(
            self._collider, self._target_vel, plan=plan,
            blackboard_queue_name=self._blackboard_queue_name, avoid_collision=False)
        # wait
        wait = DriveDistance(self.ego_vehicles[0], self._ego_distance)

        # Behavior tree
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        # root.add_child(traffic_hack)
        root.add_child(wait)
        if not Utils.NO_COLLIDER:
            root.add_child(actor_source)
            # root.add_child(actor_sink)
            root.add_child(move_actor)
            sequence.add_child(ActorTransformSetter(self._collider, self._other_actor_transform))

        sequence.add_child(root)
        # sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    # def _setup_scenario_trigger(self, config):
    #     return None

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collison_criteria = CollisionTest(self.ego_vehicles[0])
        criteria.append(collison_criteria)

        return criteria

    def __del__(self):
        self._traffic_light = None
        self.remove_all_actors()
