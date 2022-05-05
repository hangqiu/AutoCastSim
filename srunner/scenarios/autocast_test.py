#!/usr/bin/env python

"""
Test Scenarios for visualization
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

class AutoCastTest(BasicScenario):

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
    _region_threshold = 20 #  was 3 m

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

        super(AutoCastTest, self).__init__("AutoCastTest",
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
        self._other_actor_transform = right_lane_wp.transform

        # more dummy trucks blocking my view, not registering into DataProvider, so no sensor setup on these trucks
        dummy_truck_waypoint = CarlaDataProvider.get_map().get_waypoint(
            self._trigger_point_transform.location).get_left_lane()
        # dummy_truck_waypoint = dummy_truck_waypoint.next(3)[0]
        dummy_truck_blueprint = CarlaActorPool.create_blueprint('vehicle.nissan.micra', 'scenario_background')
        while not dummy_truck_waypoint.is_intersection:
            # print("Trying to spawn actor at {}".format(dummy_truck_waypoint.transform.location))
            dummy_truck_transform = carla.Transform(
                carla.Location(dummy_truck_waypoint.transform.location.x, dummy_truck_waypoint.transform.location.y, 1),
                dummy_truck_waypoint.transform.rotation)

            self.ego_vehicles[0].get_world().try_spawn_actor(dummy_truck_blueprint, dummy_truck_transform)
            dummy_truck_waypoint = dummy_truck_waypoint.next(6)[0]

        self._blocking_truck_transform = carla.Transform(
            carla.Location(dummy_truck_waypoint.transform.location.x,
                           dummy_truck_waypoint.transform.location.y,
                           1),
            carla.Rotation(dummy_truck_waypoint.transform.rotation.pitch,
                           dummy_truck_waypoint.transform.rotation.yaw -10,
                           dummy_truck_waypoint.transform.rotation.roll))
        self._occluder = CarlaActorPool.request_new_actor('vehicle.carlamotors.carlacola',
                                                          self._blocking_truck_transform)
        self._occluder.set_simulate_physics(True)
        self.other_actors.append(self._occluder)

        # other vehicles as scenario car but not moving
        actor_types = ['vehicle.volkswagen.t2', 'vehicle.audi.tt', 'vehicle.tesla.model3', 'vehicle.nissan.micra', 'vehicle.carlamotors.carlacola']
        # Cars in the left
        print("Cars in the left")
        spawn_point = carla.Transform(carla.Location(x=-63, y=127.5, z=1), carla.Rotation(yaw=180))
        left_car_1 = CarlaActorPool.request_new_actor('vehicle.bmw.grandtourer', spawn_point)
        self.other_actors.append(left_car_1)
        spawn_point = carla.Transform(carla.Location(x=-63, y=131.5, z=1), carla.Rotation(yaw=180))
        left_car_2 = CarlaActorPool.request_new_actor('vehicle.nissan.micra', spawn_point)
        self.other_actors.append(left_car_2)

        # # Cars in the right
        print("Cars in the right")
        spawn_point = carla.Transform(carla.Location(x=-100, y=136, z=1), carla.Rotation(yaw=0))
        right_car_1 = CarlaActorPool.request_new_actor('vehicle.chevrolet.impala', spawn_point)
        self.other_actors.append(right_car_1)
        spawn_point = carla.Transform(carla.Location(x=-100, y=140, z=1), carla.Rotation(yaw=0))
        right_car_2 = CarlaActorPool.request_new_actor('vehicle.audi.tt', spawn_point)
        self.other_actors.append(right_car_2)

        # # Cars in the top
        print("Cars in the top")
        spawn_point = carla.Transform(carla.Location(x=-79, y=142, z=1), carla.Rotation(yaw=270-10))
        top_car_1_1 = CarlaActorPool.request_new_actor('vehicle.carlamotors.carlacola', spawn_point)
        self.other_actors.append(top_car_1_1)
        spawn_point = carla.Transform(carla.Location(x=-78, y=150, z=1), carla.Rotation(yaw=270))
        top_car_1_2 = CarlaActorPool.request_new_actor('vehicle.tesla.model3', spawn_point)
        self.other_actors.append(top_car_1_2)
        spawn_point = carla.Transform(carla.Location(x=-78, y=157, z=1), carla.Rotation(yaw=270))
        top_car_1_3 = CarlaActorPool.request_new_actor('vehicle.volkswagen.t2', spawn_point)
        self.other_actors.append(top_car_1_3)
        #
        print("Cars in the top 2")
        spawn_point = carla.Transform(carla.Location(x=-74, y=131, z=1), carla.Rotation(yaw=270))
        top_car_2_1 = CarlaActorPool.request_new_actor('vehicle.dodge_charger.police', spawn_point)
        self.other_actors.append(top_car_2_1)
        print("Cars in the top 2")
        spawn_point = carla.Transform(carla.Location(x=-74, y=140, z=1), carla.Rotation(yaw=270))
        top_car_2_2 = CarlaActorPool.request_new_actor('vehicle.jeep.wrangler_rubicon', spawn_point)
        self.other_actors.append(top_car_2_2)
        print("Cars in the top 2")
        spawn_point = carla.Transform(carla.Location(x=-74, y=148, z=1), carla.Rotation(yaw=270))
        top_car_2_3 = CarlaActorPool.request_new_actor('vehicle.mercedes-benz.coupe', spawn_point)
        self.other_actors.append(top_car_2_3)
        print("Cars in the top 2")
        spawn_point = carla.Transform(carla.Location(x=-88, y=148, z=1), carla.Rotation(yaw=90))
        top_car_3 = CarlaActorPool.request_new_actor('vehicle.toyota.prius', spawn_point)
        self.other_actors.append(top_car_3)

        no_peds = False
        if no_peds:
            return

        # add more peds
        peds_type = 'walker.pedestrian.0001'
        # blueprintsWalkers = CarlaActorPool._blueprint_library.filter("walker.pedestrian.*")
        # print(blueprintsWalkers)
        # peds on right
        print("peds in the right")
        spawn_point = carla.Transform(carla.Location(x=-94, y=114, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0001', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-95, y=117, z=1), carla.Rotation(yaw=270))
        CarlaActorPool.request_new_actor('walker.pedestrian.0002', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-94, y=120, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0003', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-94, y=123, z=1), carla.Rotation(yaw=270))
        CarlaActorPool.request_new_actor('walker.pedestrian.0004', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-95, y=124, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0005', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-94, y=126, z=1), carla.Rotation(yaw=270))
        CarlaActorPool.request_new_actor('walker.pedestrian.0006', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-95, y=130, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0007', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-94, y=133, z=1), carla.Rotation(yaw=270))
        CarlaActorPool.request_new_actor('walker.pedestrian.0008', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-95, y=136, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0009', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-94, y=138, z=1), carla.Rotation(yaw=270))
        CarlaActorPool.request_new_actor('walker.pedestrian.0010', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-95, y=141, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0011', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-94, y=142, z=1), carla.Rotation(yaw=270))
        CarlaActorPool.request_new_actor('walker.pedestrian.0012', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-95, y=144, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0013', spawn_point)

        # peds on left
        spawn_point = carla.Transform(carla.Location(x=-68, y=115, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0014', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-67, y=116, z=1), carla.Rotation(yaw=270))
        CarlaActorPool.request_new_actor('walker.pedestrian.0015', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-68, y=119, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0016', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-67, y=124, z=1), carla.Rotation(yaw=270))
        CarlaActorPool.request_new_actor('walker.pedestrian.0017', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-68, y=126, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0018', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-67, y=128, z=1), carla.Rotation(yaw=270))
        CarlaActorPool.request_new_actor('walker.pedestrian.0019', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-68, y=131, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0020', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-67, y=134, z=1), carla.Rotation(yaw=270))
        CarlaActorPool.request_new_actor('walker.pedestrian.0021', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-68, y=138, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0022', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-67, y=140, z=1), carla.Rotation(yaw=270))
        CarlaActorPool.request_new_actor('walker.pedestrian.0023', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-68, y=141, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0024', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-67, y=145, z=1), carla.Rotation(yaw=270))
        CarlaActorPool.request_new_actor('walker.pedestrian.0025', spawn_point)
        spawn_point = carla.Transform(carla.Location(x=-68, y=147, z=1), carla.Rotation(yaw=90))
        CarlaActorPool.request_new_actor('walker.pedestrian.0026', spawn_point)



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

        ego_drive_distance = DriveDistance(self.ego_vehicles[0], self._ego_vehicle_driven_distance)
        scenario_sequence = py_trees.composites.Sequence("AutoCasTest")
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

