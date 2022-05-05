#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to CARLA scenario_runner

This is the main script to be executed when running a scenario.
It loads the scenario configuration, loads the scenario and manager,
and finally triggers the scenario execution.
"""

from __future__ import print_function

import random
import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import inspect
import os
import re
import signal
import sys
import time
import pkg_resources
import math
from copy import deepcopy
import numpy as np
import carla
import pygame


from AVR import Utils
from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
from srunner.scenariomanager.scenario_manager import ScenarioManager
from AVR.DataLogger import DataLogger
# pylint: disable=unused-import
# For the following includes the pylint check is disabled, as these are accessed via globals()
from srunner.scenarios.control_loss import ControlLoss
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle, FollowLeadingVehicleWithObstacle
from srunner.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection
from srunner.scenarios.no_signal_junction_crossing import NoSignalJunctionCrossing
from srunner.scenarios.object_crash_intersection import VehicleTurningRight, VehicleTurningLeft
from srunner.scenarios.object_crash_vehicle import StationaryObjectCrossing, DynamicObjectCrossing
from srunner.scenarios.opposite_vehicle_taking_priority import OppositeVehicleRunningRedLight
from srunner.scenarios.other_leading_vehicle import OtherLeadingVehicle
from srunner.scenarios.signalized_junction_left_turn import SignalizedJunctionLeftTurn
from srunner.scenarios.signalized_junction_right_turn import SignalizedJunctionRightTurn
from srunner.scenarios.change_lane import ChangeLane
from srunner.scenarios.cut_in import CutIn
# pylint: enable=unused-import
from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarios.route_scenario import RouteScenario
from srunner.tools.scenario_config_parser import ScenarioConfigurationParser
from srunner.tools.route_parser import RouteParser

# from AVR.backups.HUD import HUD
from AVR import HUD
# Version of scenario_runner
VERSION = 0.9


class ScenarioRunner(object):

    """
    This is the core scenario runner module. It is responsible for
    running (and repeating) a single scenario or a list of scenarios.

    Usage:
    scenario_runner = ScenarioRunner(args)
    scenario_runner.run()
    del scenario_runner
    """

    # Tunable parameters
    client_timeout = 20.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    # CARLA world and scenario handlers
    world = None
    manager = None

    additional_scenario_module = None

    agent_instance = None
    module_agent = None

    def __init__(self, args, prefix=''):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.ego_vehicles = []
        
        self._args = args
        # AVR
        arguments = self._args
        Utils.mqtt_port = arguments.mqttport
        Utils.parse_config_flags(arguments)
        if arguments.eval:
            Utils.EvalEnv.parse_eval_args(arguments.eval)

        Utils.BACKGROUND_TRAFFIC = arguments.bgtraffic
        
        record_args = deepcopy(arguments)
        Utils.environment_config = record_args

        print("&&&&&&&&&&&&&&&&&&&&Initialization")
        print("Util.mqtt_port:",Utils.mqtt_port)

        if args.timeout:
            self.client_timeout = float(args.timeout)

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        self.client.set_timeout(self.client_timeout)

        print("Traffic manager port {}".format(self._args.trafficManagerPort))
        self.traffic_manager = self.client.get_trafficmanager(int(self._args.trafficManagerPort))

        dist = pkg_resources.get_distribution("carla")
        if LooseVersion(dist.version) < LooseVersion('0.9.8'):
            raise ImportError("CARLA version 0.9.8 or newer required. CARLA version found: {}".format(dist))

        # Load additional scenario definitions, if there are any
        # If something goes wrong an exception will be thrown by importlib (ok here)
        if self._args.additionalScenario != '':
            module_name = os.path.basename(args.additionalScenario).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.additionalScenario))
            self.additional_scenario_module = importlib.import_module(module_name)

        # Load agent if requested via command line args
        # If something goes wrong an exception will be thrown by importlib (ok here)
        if self._args.agent is not None:
            module_name = os.path.basename(args.agent).split('.')[0]
            sys.path.insert(0, os.path.dirname(args.agent))
            self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(self._args.route, self._args.debug, self._args.timeout, recording=Utils.HUDLOG,sharing=args.sharing, prefix=prefix)

        # Create signal handler for SIGINT
        self._shutdown_requested = False
        # Reload config on SIGHUP (UNIX only)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._start_wall_time = datetime.now()
       


    def destroy(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup()
        if self.manager is not None:
            del self.manager
        if self.world is not None:
            del self.world
        if self.client is not None:
            del self.client

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._shutdown_requested = True
        if self.manager:
            self.manager.stop_scenario()
            self._cleanup()
            if not self.manager.get_running_status():
                raise RuntimeError("Timeout occured during scenario execution")

    def _get_scenario_class_or_fail(self, scenario):
        """
        Get scenario class by scenario name
        If scenario is not supported or not found, exit script
        """
        # print(scenario)
        # print(globals())
        if scenario in globals():
            return globals()[scenario]

        for member in inspect.getmembers(self.additional_scenario_module):
            if scenario in member and inspect.isclass(member[1]):
                return member[1]

        print("Scenario '{}' not supported ... Exiting".format(scenario))
        sys.exit(-1)

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        if self.world is not None and self.manager is not None \
                and self._args.agent and self.manager.get_running_status():
            # Reset to asynchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        self.client.stop_recorder()
        self.manager.cleanup()

        CarlaDataProvider.cleanup()
        CarlaActorPool.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                if not self._args.waitForEgo:
                    print("Destroying ego vehicle {}".format(self.ego_vehicles[i].id))
                    self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

    def _prepare_ego_vehicles(self, ego_vehicles):
        """
        Spawn or update the ego vehicles
        """

        if not self._args.waitForEgo:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaActorPool.setup_actor(vehicle.model,
                                                                    vehicle.transform,
                                                                    vehicle.rolename,
                                                                    True,
                                                                    color=vehicle.color,
                                                                    actor_category=vehicle.category))
        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # set initial speed
        # if Utils.BGTRAFFIC_INITSPD:
        for i, v in enumerate(self.ego_vehicles):
            init_spd = Utils.init_speed_mps
            """not setting ego spd as init spd for scenario ramp up"""
            # if Utils.EvalEnv.ego_speed_kmph is not None:
            #     init_spd = Utils.EvalEnv.ego_speed_kmph
            trans = v.get_transform()
            yaw = trans.rotation.yaw * (math.pi / 180)
            spd_vector = carla.Vector3D(init_spd * math.cos(yaw),
                                        init_spd * math.sin(yaw),
                                        0.0)
            self.ego_vehicles[i].set_target_velocity(spd_vector)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _analyze_scenario(self, config, trace_id=None):
        """
        Provide feedback about success/failure of a scenario
        """
        current_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        junit_filename = None
        config_name = config.name
        if self._args.outputdir != '':
            outputdir = self._args.outputdir
            if trace_id is not None:
                outputdir = os.path.join(outputdir, str(trace_id))
            config_name = os.path.join(outputdir, config_name)
        if self._args.junit:
            junit_filename = config_name + current_time + ".xml"
        filename = None
        if self._args.file:
            filename = config_name + current_time + ".txt"

        if not self.manager.analyze_scenario(self._args.output, filename, junit_filename):
            print("All scenario tests were passed successfully!")
        else:
            print("Not all scenario tests were successful")
            if not (self._args.output or filename or junit_filename):
                print("Please run with --output for further information")

    def _load_and_wait_for_world(self, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaActorPool and CarlaDataProvider
        """

        if self._args.reloadWorld:
            self.world = self.client.load_world(town)
        else:
            # if the world should not be reloaded, wait at least until all ego vehicles are ready
            ego_vehicle_found = False
            if self._args.waitForEgo:
                while not ego_vehicle_found and not self._shutdown_requested:
                    vehicles = self.client.get_world().get_actors().filter('vehicle.*')
                    for ego_vehicle in ego_vehicles:
                        ego_vehicle_found = False
                        for vehicle in vehicles:
                            if vehicle.attributes['role_name'] == ego_vehicle.rolename:
                                ego_vehicle_found = True
                                break
                        if not ego_vehicle_found:
                            print("Not all ego vehicles ready. Waiting ... ")
                            time.sleep(1)
                            break

        self.world = self.client.get_world()

        if self._args.agent:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.frame_rate
            self.world.apply_settings(settings)

        self.traffic_manager.set_synchronous_mode(True)

        if self._args.eval and len(self._args.eval)==5:
            eval_seed = gen_seeds(self._args.eval, self._args.seed)
            random.seed(eval_seed)
        print("process control seed: {}".format(eval_seed)) 
        tm_seed = int(self._args.trafficManagerSeed)
        if tm_seed == 0:
            tm_seed = random.randint(0,10000)
        print("Setting traffic manager seed: {}".format(tm_seed))
        self.traffic_manager.set_random_device_seed(tm_seed)
        CarlaActorPool.set_seed(tm_seed)

        """ TM AutoCast setting """
        self.traffic_manager.set_hybrid_physics_mode(True)
        if Utils.EvalEnv.collider_speed_kmph is not None:
            speed_diff = (20.0 - Utils.EvalEnv.collider_speed_kmph) / 20.0
            self.traffic_manager.global_percentage_speed_difference(speed_diff)

        CarlaActorPool.set_client(self.client)
        CarlaActorPool.set_world(self.world)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(self._args.trafficManagerPort))

        if self._args.route:
            self.manager.set_hud_world(self.world)

        # Wait for the world to be ready
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town and CarlaDataProvider.get_map().name != "OpenDriveMap":
            print("The CARLA server uses the wrong map: {}".format(CarlaDataProvider.get_map().name))
            print("This scenario requires to use map: {}".format(town))
            return False

        return True

    def _load_and_run_scenario(self, config):
        """
        Load and run the scenario given by config
        """
        result = False
        if not self._load_and_wait_for_world(config.town, config.ego_vehicles):
            self._cleanup()
            return False

        if self._args.agent:
            agent_class_name = self.module_agent.__name__.title().replace('_', '')
            if 'Lidar' in self._args.agent:
                agent_class_name = self._args.agent.split('/')[-1]
                agent_class_name = agent_class_name.split('.')[-2]
                print(agent_class_name)
                self.agent_instance = getattr(self.module_agent, agent_class_name)(self._args.agentConfig, num_checkpoint=self._args.num_checkpoint)
                config.agent = self.agent_instance
                print("agent_class_name", agent_class_name )
            elif 'dagger' in self._args.agent:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(self._args.agentConfig, num_checkpoint=self._args.num_checkpoint, beta=self._args.beta)
                config.agent = self.agent_instance

            else:
                try:
                    self.agent_instance = getattr(self.module_agent, agent_class_name)(self._args.agentConfig)
                    config.agent = self.agent_instance
                except Exception as e:          # pylint: disable=broad-except
                    traceback.print_exc()
                    print("Could not setup required agent due to {}".format(e))
                    self._cleanup()
                    return False

        # Prepare scenario
        print("Preparing scenario: " + config.name)
        try:
            self._prepare_ego_vehicles(config.ego_vehicles)
            if self._args.openscenario:
                scenario = OpenScenario(world=self.world,
                                        ego_vehicles=self.ego_vehicles,
                                        config=config,
                                        config_file=self._args.openscenario,
                                        timeout=100000)
            elif self._args.route:
                scenario = RouteScenario(world=self.world,
                                         config=config,
                                         debug_mode=self._args.debug)
            else:
                scenario_class = self._get_scenario_class_or_fail(config.type)
                scenario = scenario_class(self.world,
                                          self.ego_vehicles,
                                          config,
                                          self._args.randomize,
                                          self._args.debug)
        except Exception as exception:                  # pylint: disable=broad-except
            print("The scenario cannot be loaded")
            traceback.print_exc()
            print(exception)
            self._cleanup()
            return False

        # Set the appropriate weather conditions
        self.world.set_weather(config.weather)

        # Set the appropriate road friction
        if config.friction is not None:
            friction_bp = self.world.get_blueprint_library().find('static.trigger.friction')
            extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
            friction_bp.set_attribute('friction', str(config.friction))
            friction_bp.set_attribute('extent_x', str(extent.x))
            friction_bp.set_attribute('extent_y', str(extent.y))
            friction_bp.set_attribute('extent_z', str(extent.z))

            # Spawn Trigger Friction
            transform = carla.Transform()
            transform.location = carla.Location(-10000.0, -10000.0, 0.0)
            self.world.spawn_actor(friction_bp, transform)

        try:
            # Load scenario and run it
            if self._args.record:
                self.client.start_recorder("{}/{}.log".format(os.getenv('ROOT_SCENARIO_RUNNER', "./"), config.name))
            self.manager.load_scenario(scenario, self.agent_instance)
            print("CRITIA:!", self.manager.scenario.get_criteria())
            if self._args.route:
                self.manager.set_hud_agent(self.agent_instance)
            print("!!!!@#####################", self.manager._hud.trace_id)
            trace_id = self.manager._hud.trace_id
            self.manager.run_scenario()

            # Provide outputs if required
            self._analyze_scenario(config, trace_id=trace_id)

            # Remove all actors
            scenario.remove_all_actors()

            result = True

        except Exception as e:              # pylint: disable=broad-except
            traceback.print_exc()
            print(e)
            result = False

        self._cleanup()
        return result

    def _run_scenarios(self):
        """
        Run conventional scenarios (e.g. implemented using the Python API of ScenarioRunner)
        """
        result = False
        # Setup and run the scenarios for repetition times
        for _ in range(int(self._args.repetitions)):

            # Load the scenario configurations provided in the config file
            scenario_configurations = None
            print("Load Scenario {}".format(self._args.scenario))
            scenario_config_file = ScenarioConfigurationParser.find_scenario_config(
                self._args.scenario,
                self._args.configFile)
            if scenario_config_file is None:
                print("Configuration for scenario {} cannot be found!".format(self._args.scenario))
                continuefiles = sorted(glob.glob('wandb/run-*'))[-1]

            scenario_configurations = ScenarioConfigurationParser.parse_scenario_configuration(scenario_config_file,
                                                                                               self._args.scenario)

            # Execute each configuration
            for config in scenario_configurations:
                result = self._load_and_run_scenario(config)

            self._cleanup()
        return result

    def _run_route(self):
        """
        Run the route scenario
        """
        result = False
        repetitions = self._args.repetitions

        if self._args.route:
            routes = self._args.route[0]
            scenario_file = self._args.route[1]
            single_route = None
            if len(self._args.route) > 2:
                single_route = self._args.route[2]

        # retrieve routes
        route_descriptions_list = RouteParser.parse_routes_file(routes, single_route)
        # find and filter potential scenarios for each of the evaluated routes
        # For each of the routes and corresponding possible scenarios to be evaluated.

        for _, route_description in enumerate(route_descriptions_list):
            for _ in range(int(repetitions)):

                config = RouteScenarioConfiguration(route_description, scenario_file)
                result = self._load_and_run_scenario(config)

                self._cleanup()
        return result

    def _run_openscenario(self):
        """
        Run a scenario based on OpenSCENARIO
        """

        # Load the scenario configurations provided in the config file
        if not os.path.isfile(self._args.openscenario):
            print("File does not exist")
            self._cleanup()
            return False

        config = OpenScenarioConfiguration(self._args.openscenario, self.client)
        result = self._load_and_run_scenario(config)
        self._cleanup()
        return result

    def run(self):
        """
        Run all scenarios according to provided commandline args
        """
        print("Util.mqtt_port:",Utils.mqtt_port)
        self.bindmqtt()
        result = True
        print(self._args.route)
        if self._args.openscenario:
            result = self._run_openscenario()
        elif self._args.route:
            result = self._run_route()
        else:
            result = self._run_scenarios()

        print("No more scenarios .... Exiting")
        return result

    def bindmqtt(self):
        import os
        print("Binding Mqtt...", self._args.mqttport)
        print("Background Traffic", self._args.bgtraffic)
        os.system('mosquitto -p {} -d'.format(self._args.mqttport))

def gen_seeds(eval_params, seed=0):
    x = eval_params
    #if x[3]>20:
    #    return int(x[0]*100+x[1]*7+x[2]*20+x[3]*13+x[4]+11*seed)
    #else:
    #    return int(x[0]*100+x[1]*7+x[2]*20+x[3]+x[4]+11*seed)
    return int(x[0]*13+x[1]*97+x[2]*113+x[3]*107+x[4]*29+11*seed)

def main():
    """
    main function
    """


    parser = Utils.get_parser(VERSION=VERSION)

    arguments = parser.parse_args()
    # pylint: enable=line-too-long

    # AVR
    Utils.mqtt_port = arguments.mqttport

    Utils.parse_config_flags(arguments)

    if arguments.eval:
        Utils.EvalEnv.parse_eval_args(arguments.eval)

    if arguments.list:
        print("Currently the following scenarios are supported:")
        print(*ScenarioConfigurationParser.get_list_of_scenarios(arguments.configFile), sep='\n')
        return 1

    if not arguments.scenario and not arguments.openscenario and not arguments.route:
        print("Please specify either a scenario or use the route mode\n\n")
        parser.print_help(sys.stdout)
        return 1

    if (arguments.route and arguments.openscenario) or (arguments.route and arguments.scenario):
        print("The route mode cannot be used together with a scenario (incl. OpenSCENARIO)'\n\n")
        parser.print_help(sys.stdout)
        return 1

    if arguments.agent and (arguments.openscenario or arguments.scenario):
        print("Agents are currently only compatible with route scenarios'\n\n")
        parser.print_help(sys.stdout)
        return 1

    Utils.BACKGROUND_TRAFFIC = arguments.bgtraffic

    scenario_runner = None
    result = True

    if arguments.route:
        print("route option activated")
        pygame.init()
        pygame.font.init()

    try:
        scenario_runner = ScenarioRunner(arguments)
        result = scenario_runner.run()

    finally:
        if scenario_runner is not None:
            scenario_runner.destroy()
            del scenario_runner
        if arguments.route:
            pygame.quit()

    return not result


if __name__ == "__main__":
    sys.exit(main())
