#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Non-signalized junctions: crossing negotiation:

The hero vehicle is passing through a junction without traffic lights
And encounters another vehicle passing across the junction.
"""

import py_trees
import carla

from AVR import Utils
from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      SyncArrival,
                                                                      KeepVelocity,
                                                                      StopVehicle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, DrivenDistanceTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerRegion, DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario

# from AVR.Utils import get_geometric_linear_intersection_by_loc_and_intersection
from AVR import Utils
import random
import numpy as np



class AutoCastIntersectionRedLightViolation(BasicScenario):

    """
    Implementation class for
    'Non-signalized junctions: crossing negotiation' scenario,
    (Traffic Scenario 10).

    This is a single ego vehicle scenario
    """

    # ego vehicle parameters
    # _ego_vehicle_max_velocity = 20
    _ego_vehicle_driven_distance = 30 # was 105

    # other vehicle
    _other_actor_max_brake = 1.0
    _other_actor_target_velocity = 30/3.6 # m/s was 15, 30 makes collision happen
    _region_threshold = 25 #  was 3 m

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=30):
        """
        Setup all relevant parameters and create scenario
        """
        self._trigger_point_transform = config.trigger_points[0]
        self._other_actor_transform = None
        # Timeout of scenario in seconds
        # self.timeout = timeout
        self.timeout = 40.0  # Autocast timeout

        self._collider_distance = 40

        self._occluder = None
        self._collider = None

        if Utils.EvalEnv.collider_speed_kmph is not None:
            self._other_actor_target_velocity = Utils.EvalEnv.collider_speed_kmph / 3.6

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

        super(AutoCastIntersectionRedLightViolation, self).__init__("AutoCastIntersectionRedLightViolation",
                                                       ego_vehicles,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=False)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # make the car init in next right lane

        other_actor_wp = CarlaDataProvider.get_map().get_waypoint(
            config.other_actors[0].transform.location)
        right_lane_wp = other_actor_wp.get_right_lane()
        #First Lane
        self._other_actor_transform = right_lane_wp.transform
        #Second Lane
        #self._other_actor_transform = other_actor_wp.transform
        
        if not Utils.NO_COLLIDER:
            first_vehicle_transform = carla.Transform(
                carla.Location(config.other_actors[0].transform.location.x,
                               config.other_actors[0].transform.location.y,
                               config.other_actors[0].transform.location.z - 500),
                config.other_actors[0].transform.rotation)
            rolename = 'scenario'
            if Utils.PASSIVE_COLLIDER:
                rolename = Utils.PASSIVE_ACTOR_ROLENAME
            first_vehicle = CarlaActorPool.request_new_actor('vehicle.audi.tt', first_vehicle_transform, rolename=rolename)
            first_vehicle.set_simulate_physics(enabled=False)
            self._collider = first_vehicle
            self.other_actors.append(first_vehicle)
        

        # Create left turn stop vehicle
        # block view vehicle
        # trigger_waypoint_left_lane = CarlaDataProvider.get_map().get_waypoint(
        #     self._trigger_point_transform.location).get_left_lane()
        # ego_waypoint_left_line = trigger_waypoint_left_lane
        # blocking_truck_waypoint_at_intersection = None  # _blocked_vehcile_transform
        # while ego_waypoint_left_line:
        #     if ego_waypoint_left_line.is_intersection:
        #         blocking_truck_waypoint_at_intersection = ego_waypoint_left_line
        #         break
        #     ego_waypoint_left_line = ego_waypoint_left_line.next(2)[0]



        # more dummy trucks blocking my view, not registering into DataProvider, so no sensor setup on these trucks
        dummy_truck_waypoint = CarlaDataProvider.get_map().get_waypoint(
            self._trigger_point_transform.location).get_left_lane()
        if Utils.TEST_INTROVIDEO:
            dummy_truck_waypoint = dummy_truck_waypoint.next(2)[0]
        else:
            dummy_truck_waypoint = dummy_truck_waypoint.next(3)[0]
        dummy_obstacle_blueprint_name = ['vehicle.jeep.wrangler_rubicon', 'vehicle.tesla.cybertruck', 'vehicle.carlamotors.carlacola']

        bp_index = 0
        while not dummy_truck_waypoint.is_intersection:
            print("Trying to spawn actor at {}".format(dummy_truck_waypoint.transform.location))
            dummy_truck_transform = carla.Transform(
                carla.Location(dummy_truck_waypoint.transform.location.x, dummy_truck_waypoint.transform.location.y, 1),
                dummy_truck_waypoint.transform.rotation)

            if Utils.TEST_INTROVIDEO:
                dummy_truck_blueprint = CarlaActorPool.create_blueprint(dummy_obstacle_blueprint_name[bp_index%3],
                                                                    'scenario_background')
            else:
                dummy_truck_blueprint = CarlaActorPool.create_blueprint('vehicle.carlamotors.carlacola',
                                                                        'scenario_background')
            bp_index += 1
            self.ego_vehicles[0].get_world().try_spawn_actor(dummy_truck_blueprint, dummy_truck_transform)
            if Utils.TEST_INTROVIDEO:
                dummy_truck_waypoint = dummy_truck_waypoint.next(6.2)[0]
            else:
                dummy_truck_waypoint = dummy_truck_waypoint.next(6)[0]

        self._blocking_truck_transform = carla.Transform(
            carla.Location(dummy_truck_waypoint.transform.location.x,
                           dummy_truck_waypoint.transform.location.y,
                           1),
            carla.Rotation(dummy_truck_waypoint.transform.rotation.pitch,
                           dummy_truck_waypoint.transform.rotation.yaw,
                           dummy_truck_waypoint.transform.rotation.roll))
        print("occluder at:{}".format(self._blocking_truck_transform.location))
        self._occluder = CarlaActorPool.request_new_actor('vehicle.carlamotors.carlacola',
                                                          self._blocking_truck_transform)
        self._occluder.set_simulate_physics(True)
        print("other_actors", self.other_actors)
        self.other_actors.append(self._occluder)
        print("other_actors after", self.other_actors)


    def _create_behavior(self):
        """
        After invoking this scenario, it will wait for the user
        controlled vehicle to enter the start region,
        then make a traffic participant to accelerate
        until it is going fast enough to reach an intersection point.
        at the same time as the user controlled vehicle at the junction.
        Once the user controlled vehicle comes close to the junction,
        the traffic participant accelerates and passes through the junction.
        After 60 seconds, a timeout stops the scenario.
        """

        # Creating leaf nodes


        # assuming north south east west streets

        location_of_collision = Utils.get_geometric_linear_intersection_by_loc_and_intersection(
            self._trigger_point_transform.location,
            self._other_actor_transform.location)

        if location_of_collision is None:
            print(self._trigger_point_transform.location)
            print(self._other_actor_transform.location)
            raise RuntimeError("Intersecting point doesn't exist")
        # start_other_trigger = InTriggerRegion(
        #     self.ego_vehicles[0],
        #     -80, -70,
        #     -75, -60)
        if not Utils.NO_COLLIDER:
            sync_arrival = SyncArrival(
                self._collider, self.ego_vehicles[0],
                location_of_collision)
        
        '''
        collider_in_intersection = InTriggerRegion(
            self.ego_vehicles[0],
            location_of_collision.x-self._region_threshold, location_of_collision.x+self._region_threshold,
            location_of_collision.y-2*self._region_threshold, location_of_collision.y)#+self._region_threshold)
        '''
        if not Utils.NO_COLLIDER:
            collider_in_intersection = InTriggerRegion(
            self._collider,
            location_of_collision.x-self._region_threshold, location_of_collision.x+self._region_threshold,
            location_of_collision.y-self._region_threshold, location_of_collision.y+self._region_threshold)

        if not Utils.NO_COLLIDER:
            keep_velocity_other = KeepVelocity(
                self._collider,
                self._other_actor_target_velocity)

        # stop_other_trigger = InTriggerRegion(
        #     self.other_actors[0],
        #     -45, -35,
        #     -140, -130)

        # stop_other = StopVehicle(
        #     self.other_actors[0],
        #     self._other_actor_max_brake)

        # end_condition = InTriggerRegion(
        #     self.ego_vehicles[0],
        #     -90, -70,
        #     -170, -156
        # )
        '''
        #Remove the comment to let the red car move
        '''
        if not Utils.NO_COLLIDER:
            collider_pass_thru = DriveDistance(self._collider, self._collider_distance)

        ego_drive_distance = DriveDistance(self.ego_vehicles[0], self._ego_vehicle_driven_distance)

        # Creating non-leaf nodes
        # root = py_trees.composites.Sequence()
        scenario_sequence = py_trees.composites.Sequence("AutoCastIntersectionRedLightViolation")
        sync_arrival_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_velocity_other_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        if not Utils.NO_COLLIDER:
            scenario_sequence.add_child(ActorTransformSetter(self._collider, self._other_actor_transform))
            # scenario_sequence.add_child(start_other_trigger)
            scenario_sequence.add_child(sync_arrival_parallel)
            scenario_sequence.add_child(keep_velocity_other_parallel)

            # scenario_sequence.add_child(stop_other)
            # scenario_sequence.add_child(end_condition)
            # scenario_sequence.add_child(ActorDestroy(self.other_actors[0]))

            sync_arrival_parallel.add_child(sync_arrival)
            sync_arrival_parallel.add_child(collider_in_intersection)
            keep_velocity_other_parallel.add_child(keep_velocity_other)
            keep_velocity_other_parallel.add_child(ego_drive_distance)
            # keep_velocity_other_parallel.add_child(collider_pass_thru)
            # keep_velocity_other_parallel.add_child(stop_other_trigger)

        scenario_sequence.add_child(ego_drive_distance)

        return scenario_sequence

    def _setup_scenario_trigger(self, config):
        return None

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        # Adding checks for ego vehicle
        collision_criterion_ego = CollisionTest(self.ego_vehicles[0])
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicles[0], self._ego_vehicle_driven_distance)
        criteria.append(collision_criterion_ego)
        criteria.append(driven_distance_criterion)

        # Add approriate checks for other vehicles
        for vehicle in self.other_actors:
            collision_criterion = CollisionTest(vehicle)
            criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

