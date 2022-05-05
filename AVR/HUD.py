"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""
import os
import threading
import time

import carla
from AVR import Utils, PCProcess, Collaborator
import datetime
import math
import pygame
import numpy as np
import copy

import weakref
from carla import ColorConverter as cc

from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
from AVR.Sensors import CollisionSensor, GnssSensor, LaneInvasionSensor
from AVR.KeyboardControl import KeyboardControl
from AVR.DataLogger import DataLogger

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    """
    Under scenario manager, has agent wrapper
    """
    width = 720
    height = 720  # use same width height to align with lidar display
    def __init__(self, recording=False, debug_mode=False, prefix=''):
        
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        self.dim = (HUD.width, HUD.height)
        self.maxdim = Utils.LidarRange * 2 + 1 # set according to lidar range 50 * 2 + 1
        pygame.init()
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (HUD.width, 40), (0, HUD.height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), HUD.width, HUD.height)
        self.server_fps = 0
        self.server_frm_time = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self._client_clock = pygame.time.Clock()
        self._world = None
        self._vehicle = None
        self._collision_sensor = None
        self._lane_invasion_sensor = None
        self._gnss_sensor = None
        self._surface = None
        self._vehicle_index = 0
        self._display = pygame.display.set_mode((HUD.width, HUD.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        # self.showDummyCam = False
        self.showCamera = False
        self._agent = None
        self._agent_wrapper = None
        self.keyboard = KeyboardControl(self)
        self._recording = recording
        self._start = False
        self._quit = False
        self.trace_id = Utils.EvalEnv.get_trace_id()
    
        self.prefix = prefix

        self.human_control = Utils.default_control()
        self._destroyed = False
        
    def destroy(self):
        if not self._destroyed:
            self.compile_recordings()
            self._destroyed = True

    def compile_recordings(self):
        # debug video
        result_dir = Utils.RecordingOutput + str(self.trace_id) + Utils.DebugOutput
        hud_files = "/0_hud_info_*.jpg"
        pc_files = "/1_PC_*.jpg"
        rgb_files = "/2_RGB_*.jpg"
        Utils.compile_video(result_dir, "0_0_hud_info.mp4", [hud_files])
        Utils.compile_video(result_dir, "0_0_PC.mp4", [pc_files])
        Utils.compile_video(result_dir, "0_0_RGB.mp4", [rgb_files])
        # recording videos
        episode_dir = Utils.RecordingOutput + str(self.trace_id) + '/episode_' + str(self.trace_id).zfill(5)
        if os.path.exists(episode_dir):
            for dir in os.listdir(episode_dir):
                if "measurements" == dir or "config.json" == dir:
                    continue
                result_dir = os.path.join(episode_dir, dir)
                Utils.compile_video(result_dir, "0_" + dir + ".mp4")


    def set_world(self, world):
        # print("Setting HUD World and dummy sensor")
        self._world = world
        # bp = world.get_blueprint_library().find('sensor.camera.rgb')
        # self._dummy_sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=10, y=133, z=40),
        #                                                          carla.Rotation(pitch=-90, yaw=90)))
        # weak_self = weakref.ref(self)
        # self._dummy_sensor.listen(lambda image: HUD.refresh(weak_self))
        print("Finished HUD world setting")

    def set_agent(self, agent, agent_wrapper):
        self._agent = agent
        self._agent_wrapper = agent_wrapper

    def set_vehicle(self, vehicle):
        self._vehicle = vehicle

        # if self._collision_sensor is not None:
        #     del self._collision_sensor
        # if self._gnss_sensor is not None:
        #     del self._gnss_sensor
        # if self._lane_invasion_sensor is not None:
        #     del self._lane_invasion_sensor
        # self._collision_sensor = CollisionSensor(vehicle, HUD)
        # self._gnss_sensor = GnssSensor(vehicle)
        # self._lane_invasion_sensor = LaneInvasionSensor(vehicle, HUD)

    def next_vehicle(self):
        self._vehicle_index += 1

    def on_carla_tick(self, timestamp):

        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.server_frm_time = self._server_clock.get_time()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    # def hud_loop(self):
    #     while True:
    #         time.sleep(0.05) # s
    #         self._client_clock.tick()
    #         self.hud_update()
    #         if self._quit:
    #             return

    def hud_update(self):

        if not self._show_info:
            return

        ego_vehicle = CarlaActorPool.get_hero_actor()
        vehicles = list(CarlaActorPool.get_actor_dict().values())
        if ego_vehicle is None:
            return
        else:
            if self._vehicle_index == 0:
                self.set_vehicle(ego_vehicle)
            else:
                self._vehicle_index %= len(vehicles)
                self.set_vehicle(vehicles[self._vehicle_index])

        if self._vehicle is None:
            return

        self._quit = self.keyboard.parse_events(self._client_clock)

        if self._quit:
            self.destroy()
            return

        t = self._vehicle.get_transform()
        v = self._vehicle.get_velocity()
        a = self._vehicle.get_acceleration()
        c = self._vehicle.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        # colhist = self._collision_sensor.get_collision_history()
        # collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        # max_col = max(1.0, max(collision))
        # collision = [x / max_col for x in collision]
        # vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'FrameID:  % 20s' % self.frame_number,
            'Server:  % 16s FPS' % round(self.server_fps),
            'Server:  % 16s ms' % round(self.server_frm_time),
            'Client:  % 16s FPS' % round(self._client_clock.get_fps()),
            'Client:  % 16s ms' % round(self._client_clock.get_time()),
            'ScenStart: % 18s' % (Utils.InTriggerRegion_GlobalUtilFlag),
            '',
            'Vehicle: % 20s' % Utils.get_actor_display_name(self._vehicle, truncate=20),
            'Map:     % 20s' % self._world.get_map().name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            'Accel:   % 15.0f m/s^2' % (math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            # 'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (self._gnss_sensor.lat, self._gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']

        clean_steer = 0
        if self._agent._agent_control is not None:
            clean_steer = self._agent._agent_control.steer

        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', clean_steer, -1.0, 1.0),
                ('SteerApplied:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            # 'Collision Predicted: %s' % self._agent._agent.collision_detected,
            # '2sec Horizon Clear: %s' % self._agent._agent.trajectory_clear,
            # collision,
            # 'Collision:',
            # [],
            # '',
            'Number of vehicles: % 8d' % len(vehicles)]

        self._info_text += ["Collision Details:"]
        if hasattr(self._agent._agent, 'collision_detail') and self._agent._agent.collision_detail is not None:
            self._info_text += self._agent._agent.collision_detail.to_str_list()

        if len(vehicles) > 1:
            # print("1")
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt(
                (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != self._vehicle.id]
            # print("2")

            vehicles.sort(key = lambda x: x[0])
            for d, vehicle in vehicles:
                if d > 100.0:
                    break
                vehicle_type = Utils.get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% .2fm %s' % (d, vehicle_type))
                v_spd = vehicle.get_velocity()
                v_accel = vehicle.get_acceleration()
                self._info_text.append('% .2fm, %.1fkm/h, %.1fm/s^2' % (PCProcess.get_min_corner_distance(self._vehicle, vehicle),
                                                                       3.6 * math.sqrt(v_spd.x ** 2 + v_spd.y ** 2 + v_spd.z ** 2),
                                                                       math.sqrt(v_accel.x ** 2 + v_accel.y ** 2 + v_accel.z ** 2)))

        # print(self._info_text)
        self._notifications.tick(self._world, self._client_clock)
        self.render()

        # self.refresh()
        pygame.display.flip()
        
        # added for recording
        # if Utils.InTriggerRegion_GlobalUtilFlag:
        try:
            if self._agent._expert_control is not None:
                c = self._agent._expert_control
        except:
            pass
        DataLogger.record_sensor_dataset(self.frame_number, ego_action=c, agent_wrapper=self._agent_wrapper,
                                            route_id=self.trace_id, ego_vehicle_id=ego_vehicle.id)

    def tick(self):
        self.hud_update()
        return self._quit

    def toggle_info(self):
        self._show_info = not self._show_info

    def toggle_camera(self):
        self.showCamera = not self.showCamera

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self):

        if not self:
            return

        if self._vehicle is None:
            return

        """ data surface """
        data_obj = self._agent.sensor_interface.get_data_obj()
        ego_sensor_id = str(self._vehicle.id) + "_RGB"
        ego_rgb_obj = data_obj[ego_sensor_id][1]

        object_list = []
        collaborator = self._agent_wrapper.get_collaborator_for_hud(self._vehicle.id)
        if collaborator is None:
            # return
            # raise Exception('Collaborator None')
            ego_lidar_id = str(self._vehicle.id) + Collaborator.LidarSensorName
            lidar = self._agent.sensor_interface.get_data_by_id(ego_lidar_id)
            pc_raw = copy.deepcopy(lidar)
            shared_sensor_index = len(pc_raw)
        else:
            pc_raw = copy.deepcopy(collaborator.fused_sensor_data)
            shared_sensor_index = collaborator.ego_lidar_ending_index
            object_list = collaborator.drawing_object_list
        # print("HUD lidar shape {}".format(pc_raw.shape))
        pc = Utils.lidar_to_hud_image(pc_raw, self.dim, self.maxdim, shared_sensor_index)

        ego_rgb_obj.convert(cc.Raw)
        array = np.frombuffer(ego_rgb_obj.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (ego_rgb_obj.height, ego_rgb_obj.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        _pc_surface = pygame.surfarray.make_surface(pc)
        _rgb_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self._recording:
            image_name = Utils.RecordingOutput + str(self.trace_id) + Utils.DebugOutput + "/1_PC_{}.jpg".format(
                self.frame_number)
            pygame.image.save(_pc_surface, image_name)  # image only
            image_name = Utils.RecordingOutput + str(
                self.trace_id) + Utils.DebugOutput + "/2_RGB_{}.jpg".format(self.frame_number)
            pygame.image.save(_rgb_surface, image_name)  # image only

        if self.showCamera or Utils.HUMAN_AGENT:
            self._surface = _rgb_surface
        else:
            self._surface = _pc_surface

        # drawing trajectory
        # try:

        waypoints = self._agent.agent_trajectory_points_timestamp
        predict_trajectory_list = self._agent.collider_trajectory_points_timestamp
        myTrans = self._vehicle.get_transform()
        self.draw_boundingbox(object_list)
        # draw predicted object trajectory
        if predict_trajectory_list is not None:
            for predict_trajectory in predict_trajectory_list:
                if len(predict_trajectory) > 0:
                    self.draw_trajectory(predict_trajectory, myTrans, self.frame_number)
        # draw ego planned path
        if len(waypoints) > 0:
            self.draw_trajectory(waypoints, myTrans, self.frame_number)

        # trainee planned path
        if hasattr(self._agent, '_trainee_planned_path'):
            if self._agent._trainee_planned_path is not None:
                self.draw_path(self._agent._trainee_planned_path, myTrans)

        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))

        """ show info surface """
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            self._display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(self._display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(self._display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(self._display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        try:
                            if item[2] < 0.0:
                                rect = pygame.Rect(bar_h_offset + f * (bar_width - 6), v_offset + 8, 6, 6)
                            else:
                                rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        except TypeError:
                            print("Invalid value: {},{}".format(item[0], item[1]))
                            pass
                        pygame.draw.rect(self._display, (255, 255, 255), rect)
                    item = item[0]
                if item: # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    self._display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(self._display)
        self.help.render(self._display)

        if self._recording:
            image_name = Utils.RecordingOutput + str(
                self.trace_id) + Utils.DebugOutput + "/0_hud_info_{}.jpg".format(self.frame_number)
            pygame.image.save(self._display, image_name)  # with info text

    def draw_path(self, path, myTrans):
        n_points = len(path)
        path = np.array(path)
        path = self.path_to_HUD_view(path, myTrans)
        for i in range(n_points):
            pygame.draw.circle(self._surface, [255, n_points*i, n_points*i],
                                   path[i], 5)

    def draw_trajectory(self, waypoints, myTrans, cur_frame_id, sample_step=1):
        sampled_waypoints = []
        for index, point in enumerate(waypoints):
            if index % sample_step != 0:
                continue
            if index < 3: # for visualizing shared points
                continue
            sampled_waypoints.append(point)

        sampled_waypoints = np.array(sampled_waypoints)
        timestamp = sampled_waypoints[:, 3]
        sampled_waypoints_xyz = sampled_waypoints[:, :3]
        sampled_waypoints_xyz = self.path_to_HUD_view(sampled_waypoints_xyz, myTrans)

        color_output = self.gradient_color_based_on_timestamp(timestamp, cur_frame_id)
        # print(color_output)
        # print(sampled_waypoints)
        for i in range(len(sampled_waypoints)):
            pygame.draw.circle(self._surface, color_output[i],
                                   sampled_waypoints_xyz[i], 5)

    def path_to_HUD_view(self, sampled_waypoints_xyz, myTrans):

        sampled_waypoints_xyz = Utils.world_to_car_transform(sampled_waypoints_xyz, myTrans)

        sampled_waypoints_xyz = sampled_waypoints_xyz[:, :2]
        sampled_waypoints_xyz *= min(self.dim) / self.maxdim

        sampled_waypoints_xyz += (0.5 * self.dim[0], 0.5 * self.dim[1])
        # new_waypoints = np.fabs(new_waypoints)
        sampled_waypoints_xyz = sampled_waypoints_xyz.astype(np.int32)
        sampled_waypoints_xyz = np.reshape(sampled_waypoints_xyz, (-1, 2))
        return sampled_waypoints_xyz


    def gradient_color_based_on_timestamp(self, timestamp, cur_frame_time):
        """# gradient color rule: within 255 frames, G -> B, after 255 frames, B """
        color_output = []
        # print(timestamp)
        # print(timestamp[0])
        for time in timestamp:
            delta_time = (time-cur_frame_time) * 5
            if 0 < delta_time < 255.0:
                color_output.append([0, 255-delta_time, delta_time])
            elif delta_time >= 255.0:
                color_output.append([0, 0, 255])
            else:
                color_output.append([0, 255, 255])
        return color_output

    def gradient_color_based_on_index(self, sampled_waypoints):
        R = 0
        G = 255
        B = 0
        delta = 10
        G2B = True
        color_output = []
        for index, point in enumerate(sampled_waypoints):
            if G == 0:
                G2B = False
            if B == 0:
                G2B = True
            if G2B:
                G = max(0, G - delta)
                B = min(255, B + delta)
            else:
                G = min(255, G + delta)
                B = max(0, B - delta)
            color_output.append([R, G, B])
        return color_output


    # def draw_trajectory(self, waypoints, predict_trajectory, myTrans, sample_step=1):
    #     if len(waypoints) > 2:
    #         new_waypoints = np.matrix(waypoints)
    #         new_waypoints = new_waypoints[:, :3]
    #         new_waypoints = Utils.world_to_car_transform(new_waypoints, myTrans)
    #
    #         new_waypoints = new_waypoints[:, :2]
    #         new_waypoints *= min(self.dim) / self.maxdim
    #
    #         new_waypoints += (0.5 * self.dim[0], 0.5 * self.dim[1])
    #         new_waypoints = np.fabs(new_waypoints)
    #         new_waypoints = new_waypoints.astype(np.int32)
    #         new_waypoints = np.reshape(new_waypoints, (-1, 2))
    #         # pygame.draw.lines(self._surface, [0, 255, 0], False, new_waypoints, 2)
    #
    #         for index, point in enumerate(new_waypoints):
    #             if index + 1 != new_waypoints.shape[0] and index % sample_step == 0:
    #                 pygame.draw.circle(self._surface, [(index * 5) % 256, (255 - index * 10) % 256, (index * 7) % 256],
    #                                    new_waypoints[index], 5)
    #                 # pygame.draw.line(self._surface, [255-index*3, 0, index*3], new_waypoints[index], new_waypoints[index], 10)
    #
    #     if predict_trajectory is not None and len(predict_trajectory) > 1:
    #         predict_trajectory = predict_trajectory[1:]
    #         new_waypoints = np.matrix(predict_trajectory)
    #         new_waypoints = new_waypoints[:, :3]
    #         new_waypoints = Utils.world_to_car_transform(new_waypoints, myTrans)
    #
    #         new_waypoints = new_waypoints[:, :2]
    #         new_waypoints *= min(self.dim) / self.maxdim
    #
    #         new_waypoints += (0.5 * self.dim[0], 0.5 * self.dim[1])
    #         new_waypoints = np.fabs(new_waypoints)
    #         new_waypoints = new_waypoints.astype(np.int32)
    #         new_waypoints = np.reshape(new_waypoints, (-1, 2))
    #         # pygame.draw.lines(self._surface, [255, 0, 0], False, new_waypoints, 2)
    #         for index, point in enumerate(new_waypoints):
    #             if index + 1 != new_waypoints.shape[0] and index % sample_step == 0:
    #                 pygame.draw.circle(self._surface, [(index * 5) % 256, (255 - index * 10) % 256, (index * 7) % 256],
    #                                    new_waypoints[index], 5)
    #                 # pygame.draw.line(self._surface, [255-index*3, 0, index*3], new_waypoints[index], new_waypoints[index], 10)

    def draw_boundingbox(self, object_list):
        for object in object_list:
            coordinate = object[:, :2]
            coordinate *= min(self.dim) / self.maxdim
            coordinate += (0.5 * self.dim[0], 0.5 * self.dim[1])
            coordinate = np.fabs(coordinate)
            coordinate = coordinate.astype(np.int32)
            coordinate = np.reshape(coordinate, (-1, 2))
            pygame.draw.polygon(self._surface, [243, 156, 18], coordinate, 1)



