#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario spawning elements to make the town dynamic and interesting
"""

from srunner.scenariomanager.carla_data_provider import CarlaActorPool
from srunner.scenarios.basic_scenario import BasicScenario

from AVR import Utils
import math
import carla


class BackgroundActivity(BasicScenario):

    """
    Implementation of a scenario to spawn a set of background actors,
    and to remove traffic jams in background traffic

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, timeout=35 * 60):
        """
        Setup all relevant parameters and create scenario
        """
        self.config = config
        self.debug = debug_mode

        self.timeout = timeout  # Timeout of scenario in seconds

        super(BackgroundActivity, self).__init__("BackgroundActivity",
                                                 ego_vehicles,
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 terminate_on_failure=True,
                                                 criteria_enable=True)

    def _initialize_actors(self, config):
        for actor in config.other_actors:
            new_actors = CarlaActorPool.request_new_batch_actors(actor.model,
                                                                 actor.amount,
                                                                 actor.transform,
                                                                 hero=False,
                                                                 autopilot=actor.autopilot,
                                                                 random_location=actor.random_location)
            if new_actors is None:
                raise Exception("Error: Unable to add actor {} at {}".format(actor.model, actor.transform))


            for _actor in new_actors:

                if Utils.BGTRAFFIC_INITSPD:
                    # init speed
                    init_spd = Utils.init_speed_mps
                    if Utils.EvalEnv.collider_speed_kmph is not None:
                        init_spd = Utils.EvalEnv.collider_speed_kmph / 3.6
                    trans = _actor.get_transform()
                    yaw = trans.rotation.yaw * (math.pi / 180)
                    spd_vector = carla.Vector3D(init_spd * math.cos(yaw),
                                                init_spd * math.sin(yaw),
                                                0.0)
                    _actor.set_target_velocity(spd_vector)

                self.other_actors.append(_actor)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        pass

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        pass

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
