#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the Scenario and ScenarioManager implementations.
These must not be modified and are for reference only!
"""

from __future__ import print_function
import os
import sys
import time

import py_trees

from srunner.autoagents.agent_wrapper import AgentWrapper
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.timer import GameTime, TimeOut
from srunner.scenariomanager.watchdog import Watchdog

from AVR.HUD import HUD
from AVR.DataLogger import DataLogger
from AVR import Utils, Collaborator


class Scenario(object):

    """
    Basic scenario class. This class holds the behavior_tree describing the
    scenario and the test criteria.

    The user must not modify this class.

    Important parameters:
    - behavior: User defined scenario with py_tree
    - criteria_list: List of user defined test criteria with py_tree
    - timeout (default = 60s): Timeout of the scenario in seconds
    - terminate_on_failure: Terminate scenario on first failure
    """

    def __init__(self, behavior, criteria, name, timeout=60, terminate_on_failure=True):
        self.behavior = behavior
        self.test_criteria = criteria
        self.timeout = timeout
        self.name = name

        if self.test_criteria is not None and not isinstance(self.test_criteria, py_trees.composites.Parallel):
            # list of nodes
            for criterion in self.test_criteria:
                criterion.terminate_on_failure = terminate_on_failure

            # Create py_tree for test criteria
            self.criteria_tree = py_trees.composites.Parallel(name="Test Criteria")
            self.criteria_tree.add_children(self.test_criteria)
            self.criteria_tree.setup(timeout=1)
        else:
            self.criteria_tree = criteria

        # Create node for timeout
        self.timeout_node = TimeOut(self.timeout, name="TimeOut")

        # Create overall py_tree
        self.scenario_tree = py_trees.composites.Parallel(name, policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        if behavior is not None:
            self.scenario_tree.add_child(self.behavior)
        self.scenario_tree.add_child(self.timeout_node)
        if criteria is not None:
            self.scenario_tree.add_child(self.criteria_tree)
        self.scenario_tree.setup(timeout=1)

    def _extract_nodes_from_tree(self, tree):  # pylint: disable=no-self-use
        """
        Returns the list of all nodes from the given tree
        """
        node_list = [tree]
        more_nodes_exist = True
        while more_nodes_exist:
            more_nodes_exist = False
            for node in node_list:
                if node.children:
                    node_list.remove(node)
                    more_nodes_exist = True
                    for child in node.children:
                        node_list.append(child)

        if len(node_list) == 1 and isinstance(node_list[0], py_trees.composites.Parallel):
            return []

        return node_list

    def get_criteria(self):
        """
        Return the list of test criteria (all leave nodes)
        """
        criteria_list = self._extract_nodes_from_tree(self.criteria_tree)
        return criteria_list

    def terminate(self):
        """
        This function sets the status of all leaves in the scenario tree to INVALID
        """
        # Get list of all nodes in the tree
        node_list = self._extract_nodes_from_tree(self.scenario_tree)

        # Set status to INVALID
        for node in node_list:
            node.terminate(py_trees.common.Status.INVALID)


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, and analyze a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.execute()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. Trigger a result evaluation with manager.analyze()
    5. Cleanup with manager.stop_scenario()
    """

    def __init__(self, route_mode=True, debug_mode=False, timeout=2.0, recording=False, sharing=False, prefix=''):
        """
        Init requires scenario as input
        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = timeout
        self._watchdog = Watchdog(float(self._timeout))

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None

        self._route_mode = route_mode
        
        """AVR"""
        Utils.RecordingOutput = os.path.join(Utils.RecordingOutput, prefix) + '/'

        self._temp_hud = True
        self._hud = None
        self._hud_debug = False
        if self._route_mode and self._temp_hud:
            self._hud = HUD(recording=recording, debug_mode=self._hud_debug)
        self.sharing_session = sharing
        print("Finished initializing scenario manager")

    def _reset(self):
        """
        Reset all parameters
        """
        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        GameTime.restart()

    def cleanup(self):
        """
        This function triggers a proper termination of a scenario
        """
        if self._hud is not None:
            self._hud.destroy()
        if self.scenario is not None:
            self.scenario.terminate()
        if self._agent is not None:
            self._agent.cleanup()
            # self._agent = None

        CarlaDataProvider.cleanup()
        CarlaActorPool.cleanup()



    def load_scenario(self, scenario, agent=None):
        """
        Load a new scenario
        """
        self._reset()
        self._agent = AgentWrapper(agent) if agent else None
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors

        CarlaDataProvider.register_actors(self.ego_vehicles)
        CarlaDataProvider.register_actors(self.other_actors)
        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        if self._agent is not None:
            self._agent.setup_sensors(self.ego_vehicles[0], debug_mode=self._debug_mode)
            self._agent.setup_collaborator(self.ego_vehicles[0], self.sharing_session)
            self._agent.setup_beacon()
            if self.sharing_session:
                pass

    def check_sensors_on_other_actors(self, dist_thresh=50):
        """
        Keep actors in vicinity with sensors.
        """
        vehicles = CarlaActorPool.get_actors()
        for id, vehicle in vehicles:
            """ filter ego vehicles """
            is_ego = False
            for ego_vehicle in self.ego_vehicles:
                # print("EGO ID: {}".format(ego_vehicle.id))
                if id == ego_vehicle.id:
                    is_ego = True
                    break
            if is_ego:
                continue
            if vehicle.attributes['role_name'] == Utils.PASSIVE_ACTOR_ROLENAME:
                continue
            """ decide distance """
            nearby = False
            vehicle_location = CarlaDataProvider.get_location(vehicle)
            for ego_vehicle in self.ego_vehicles:
                ego_location = CarlaDataProvider.get_location(ego_vehicle)
                if (ego_location is not None) and (vehicle_location is not None) and (ego_location.distance(vehicle_location) < dist_thresh):
                    nearby = True
                    break
            sensor_id = str(id) + Collaborator.LidarSensorName
            # print("Sensor ID: {}".format(sensor_id))
            existing_sensor = self._agent.has_sensor(sensor_id)
            if nearby and (not existing_sensor):
                # self._agent.try_setup_lidar_on_other_vehicle(vehicle, debug_mode=self._debug_mode)
                self._agent.setup_sensors(vehicle, debug_mode=self._debug_mode)
                self._agent.setup_collaborator(vehicle, self.sharing_session)
            if (not nearby) and existing_sensor:
                self._agent.destroy_sensors(id)
                self._agent.destroy_collaborator(id)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True

        while self._running:
            print("=================================================================================================")
            time_start = time.time()
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                """AVR share before action"""
                if self._route_mode and self.sharing_session:
                    self.check_sensors_on_other_actors()
                sensor_update = time.time()
                if Utils.TIMEPROFILE: print("Sensor Update: {} s, Total {} s".format(sensor_update-time_start, sensor_update-time_start))
                if self._agent is not None:
                    """parallel async version"""
                    # self._agent.join_collaborators()
                    # self._agent.tick_beacon()
                    # self._agent.tick_collaborators()

                    if not Utils.TIMEPROFILE:
                        """parallel sync version"""
                        self._agent.tick_collaborators() # at least tick ego vehicle's collaborator, even without sharing flag
                        self._agent.join_collaborators()
                        if self.sharing_session:
                            self._agent.tick_beacon()
                    else:
                        """single thread version"""
                        for c in self._agent._collaborator_dict.values():
                            if c:
                                c.tick()
                                c.tick_join()
                        if self.sharing_session:
                            self._agent.tick_beacon()
                collab_update = time.time()
                if Utils.TIMEPROFILE: print("Collaborator: {} s, Total {} s".format(collab_update - sensor_update, collab_update-time_start))

                self._tick_scenario(timestamp)
                scen_update = time.time()
                if Utils.TIMEPROFILE: print("Scenario Tick: {} s, Total {} s".format(scen_update - collab_update, scen_update - time_start))
                """AVR Visualization"""
                if self._temp_hud:
                    if self._hud.tick():
                        break
                hud_update = time.time()
                if Utils.TIMEPROFILE: print("HUD update: {} s, Total {} s".format(hud_update - scen_update, hud_update - time_start))

        self._watchdog.stop()

        self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - \
            self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario
        This function is a callback for world.on_tick()

        Important:
        - It has to be ensured that the scenario has not yet completed/failed
          and that the time moved forward.
        - A thread lock should be used to avoid that the scenario tick is performed
          multiple times in parallel.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()

            if self._debug_mode:
                print("\n--------- Tick ---------\n")

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            if self._route_mode and self._temp_hud:
                self._hud.on_carla_tick(timestamp)

            if self._agent is not None:
                ego_action = self._agent()
                if Utils.HUMAN_AGENT:
                    self.ego_vehicles[0].apply_control(self._hud.human_control)
                else:
                    self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False



        if self._agent and self._running and self._watchdog.get_status():
            CarlaDataProvider.get_world().tick()

    def get_running_status(self):
        """
        returns:
           bool:  False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function is used by the overall signal handler to terminate the scenario execution
        """
        self._running = False

    def analyze_scenario(self, stdout, filename, junit):
        """
        This function is intended to be called from outside and provide
        the final statistics about the scenario (human-readable, in form of a junit
        report, etc.)
        """

        failure = False
        timeout = False
        result = "SUCCESS"

        if self.scenario.test_criteria is None:
            return True

        for criterion in self.scenario.get_criteria():
            if (not criterion.optional and
                    criterion.test_status != "SUCCESS" and
                    criterion.test_status != "ACCEPTABLE"):
                failure = True
                result = "FAILURE"
            elif criterion.test_status == "ACCEPTABLE":
                result = "ACCEPTABLE"

        if self.scenario.timeout_node.timeout and not failure:
            timeout = True
            result = "TIMEOUT"

        output = ResultOutputProvider(self, result, stdout, filename, junit)
        output.write()

        return failure or timeout

    def set_hud_agent(self, agent):
        if self._hud:
            self._hud.set_agent(agent, self._agent)

    def set_hud_world(self, world):
        if self._hud:
            self._hud.set_world(world)
