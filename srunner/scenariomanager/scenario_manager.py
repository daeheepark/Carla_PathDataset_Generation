#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementation.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import sys
import time
import copy
import math
import pygame
from PIL import Image

import py_trees

from srunner.autoagents.agent_wrapper import AgentWrapper
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from srunner.scenariomanager.utils_pdh import *
from collections import deque, defaultdict

CROP = 224
Sampling_Intv = 10
PAST_LEN = 4
FUTURE_LEN = 6

class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, and analyze a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. Trigger a result evaluation with manager.analyze_scenario()
    5. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, debug_mode=False, sync_mode=False, timeout=2.0):
        """
        Setups up the parameters, which will be filled at load_scenario()

        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._sync_mode = sync_mode
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = timeout
        self._watchdog = Watchdog(float(self._timeout))

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None

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

        if self.scenario is not None:
            self.scenario.terminate()

        if self._agent is not None:
            self._agent.cleanup()
            self._agent = None

        CarlaDataProvider.cleanup()

    def load_scenario(self, scenario, agent=None):
        """
        Load a new scenario
        """
        self._reset()
        self._agent = AgentWrapper(agent) if agent else None
        if self._agent is not None:
            self._sync_mode = True
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.town = scenario.config.town

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        if self._agent is not None:
            self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        world = CarlaDataProvider.get_world()
        map_img = Image.open(f'./maps/{self.town}.tga')
        margin = 50
        mwaypoints = world.get_map().generate_waypoints(2)
        max_x = max(mwaypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(mwaypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(mwaypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(mwaypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        _width = max(max_x - min_x, max_y - min_y)
        _world_offset = (min_x, min_y)
        lr_threshold = (0.01, 0.2)

        actor_list = world.get_actors().filter('vehicle.*')
        # pasts_and_futures = {'ego': defaultdict(deque), 'background' : defaultdict(deque)}
        pasts_and_futures = defaultdict(defaultdict)
        pasts_and_futures['past_length'] = PAST_LEN
        pasts_and_futures['future_length'] = FUTURE_LEN
        for actor in actor_list:
            pasts_and_futures['pasts'][actor.id] = deque(PAST_LEN*[0,0], maxlen=PAST_LEN)
            pasts_and_futures['futures'][actor.id] = deque(FUTURE_LEN*[0,0], maxlen=FUTURE_LEN)
            
        start_frame = world.get_snapshot().frame

        while self._running:
            timestamp = None
            # world = CarlaDataProvider.get_world()

            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp

            spectator = world.get_spectator()
            spec_transform = snapshot.find(CarlaDataProvider.get_hero_actor().id).get_transform()
            spec_transform.location.z = 40
            spec_transform.rotation.pitch = -90
            spec_transform.rotation.yaw = 0
            spec_transform.rotation.roll = 0
            spectator.set_transform(spec_transform)

            hero_transform = snapshot.find(CarlaDataProvider.get_hero_actor().id).get_transform()
            
            left_or_right_decision = 0
            if (timestamp.frame - start_frame) % Sampling_Intv == 0:
                left_or_right_1 = (hero_transform.location.x, hero_transform.location.x, hero_transform.rotation.yaw)
            if (timestamp.frame - start_frame) % Sampling_Intv == (Sampling_Intv-1): 
                left_or_right_5 = (hero_transform.location.x, hero_transform.location.x, hero_transform.rotation.yaw)
                delta_x = left_or_right_5[0] - left_or_right_1[0]
                delta_y = left_or_right_5[1] - left_or_right_1[1]
                delta_yaw = left_or_right_5[2] - left_or_right_1[2]

                if delta_x * delta_y > lr_threshold[0] and delta_yaw > lr_threshold[1]:
                    left_or_right_decision = 1
                elif delta_x * delta_y > lr_threshold[0] and delta_yaw < -lr_threshold[1]:
                    left_or_right_decision = -1
                elif abs(delta_x) < 0.0001 and abs(delta_y) < 0.0001:
                    left_or_right_decision = -2
                else:
                    left_or_right_decision = 0

            current_df = get_df(self.scenario.name, self.town, world, pasts_and_futures, start_frame, timestamp, hero_transform, 
            map_img, _world_offset, left_or_right_decision, Sampling_Intv)
            if timestamp:
                self._tick_scenario(timestamp)

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
        Run next tick of scenario and the agent.
        If running synchornously, it also handles the ticking of the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()

            if self._debug_mode:
                print("\n--------- Tick ---------\n")

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            if self._agent is not None:
                ego_action = self._agent()

            if self._agent is not None:
                self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

        if self._sync_mode and self._running and self._watchdog.get_status():
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

    def analyze_scenario(self, stdout, filename, junit, json):
        """
        This function is intended to be called from outside and provide
        the final statistics about the scenario (human-readable, in form of a junit
        report, etc.)
        """

        failure = False
        timeout = False
        result = "SUCCESS"

        if self.scenario.test_criteria is None:
            print("Nothing to analyze, this scenario has no criteria")
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

        output = ResultOutputProvider(self, result, stdout, filename, junit, json)
        output.write()

        return failure or timeout
