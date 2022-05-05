# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
import argparse
import copy

import carla
import re
import pygame
import math
import os
import cv2
import numpy as np
import glob
from time import time
from numpy.linalg import norm
from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
from tqdm import tqdm
from numba import jit

###########################
##########Constants########
###########################


class Utils:

    AgentDebug = False
    EmulationMode = False
    DistributedMode = False
    scalability_eval = False
    QuantizationGrid = 10.0
    mqtt_port = 1883
    CarlaFPS = 20.0
    View = 8
    HalfRadioRange = 50
    TimeSlot = 10 #ms
    SlotPerSession = 400
    Rate = 7.2 #was 7.2 LTE-Direct 10MHz 3.6 DSRC 10MHz
    V2XRate = 50  # was 7.2 LTE-Direct 10MHz 3.6 DSRC 10MHz
    AutoPeriod = 60
    target_speed_kmph = 20
    target_speed_mps = 5.56
    LidarRoofTopDistance = 0.5
    LidarRoofForwardDistance = 0 # was 0.7
    CameraRoofTopDistance = -0.3
    CameraRoofForwardDistance = 1.3
    CamWidth = 1280 # was 300
    CamHeight = 720 # was 200
    # vicinity threshold for waypoint eviction
    RecordingOutput = './result/'
    DebugOutput = '/BirdeyeViewDebug/'
    LidarRange = 50.0 # meters
    drivable_lane_type = carla.LaneType.Driving | carla.LaneType.Bidirectional
    perception_lane_type = drivable_lane_type | carla.LaneType.Shoulder | carla.LaneType.Parking | carla.LaneType.Sidewalk

    DEVIATION_WEIGHT = 1.0
    BACKGROUND_TRAFFIC = None
    LidarYawCorrection = 90.0
    init_speed_mps = 5.0 / 3.6 # m/s

    TEST_INTROVIDEO = False
    ###########################
    ##########Flags############
    ###########################
    # SHARING_SESSION = False
    # autopilot_mode = False
    SlicedView = True
    TRACELOG = False # Record data and schedule for radio experiment playback, turn off for performance
    COMMLOG = False # Turn off for performance, must be true if LOGCOVERAGE IS TRUE
    LOGCOVERAGE = False # LOG must be true to function
    DATALOG = False # Turn off for performance
    LEANDATALOG = False
    DAGGERDATALOG = False
    HUDLOG = False

    # autocast_mode = False #True for selective view transmission, False for all full View
    object_oriented_sharing = True
    # object_oriented_sharing = False



    AGNOSTIC = False # agnostic scheduling baseline
    VORONOI = False # EMP voronoi baseline
    # source = (0, -3, 0)
    destination = (0, 0, 0)
    
    TIMEPROFILE = False
    
    InTriggerRegion_GlobalUtilFlag = True

    Fast_Lidar = False # True mode is not working well
    Extrapolation = False
    # Fairness_Compensation = False

    PATH_REUSE = False
    NO_COLLIDER = False
    PASSIVE_COLLIDER = False # collider doesn't have sharing capability
    BGTRAFFIC_INITSPD = False

    HUMAN_AGENT=False

    PASSIVE_ACTOR_ROLENAME = 'passive'

    @staticmethod
    def get_parser(VERSION):
        description = ("AutoCast Config: Setup, Run and Evaluate scenarios using CARLA Scenario Runner")
        parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--host', default='127.0.0.1',
                            help='IP of the host server (default: localhost)')
        parser.add_argument('--port', default='2000',
                            help='TCP port to listen to (default: 2000)')
        parser.add_argument('--trafficManagerPort', default='8000',
                            help='Port to use for the TrafficManager (default: 8000)')
        parser.add_argument('--trafficManagerSeed', default='0',
                            help='Random seed used by the TrafficManager to config background traffic (default: 0)')
        parser.add_argument('--debug', action="store_true", help='Run with debug output')
        parser.add_argument('--output', action="store_true", help='Provide results on stdout')
        parser.add_argument('--file', action="store_true", help='Write results into a txt file')
        parser.add_argument('--junit', action="store_true", help='Write results into a junit file')
        parser.add_argument('--waitForEgo', action="store_true", help='Connect the scenario to an existing ego vehicle')
        parser.add_argument('--configFile', default='',
                            help='Provide an additional scenario configuration file (*.xml)')
        parser.add_argument('--additionalScenario', default='',
                            help='Provide additional scenario implementations (*.py)')
        parser.add_argument('--reloadWorld', action="store_true",
                            help='Reload the CARLA world before starting a scenario (default=True)')
        # pylint: disable=line-too-long
        parser.add_argument(
            '--scenario',
            help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle')
        parser.add_argument('--randomize', action="store_true", help='Scenario parameters are randomized')
        parser.add_argument('--repetitions', default=1, help='Number of scenario executions')
        parser.add_argument('--list', action="store_true", help='List all supported scenarios and exit')
        parser.add_argument(
            '--agent',
            help="Agent used to execute the scenario (optional). Currently only compatible with route-based scenarios.")
        parser.add_argument('--agentConfig', type=str, help="Path to Agent's configuration file", default="")
        parser.add_argument('--openscenario', help='Provide an OpenSCENARIO definition')
        parser.add_argument(
            '--route', help='Run a route as a scenario (input: (route_file,scenario_file,[number of route]))',
            nargs='+', type=str)
        parser.add_argument('--record', action="store_true",
                            help='Use CARLA recording feature to create a recording of the scenario')
        parser.add_argument('--commlog', action="store_true",
                            help='Record communication logs among vehicles')
        parser.add_argument('--full', action="store_true",
                            help='Record full records(RGB, Lidar, FusedLidar) for data logger')
        parser.add_argument('--daggerdatalog', action="store_true",
                            help='Record only Lidar for every vehicles')
        parser.add_argument('--lean', action="store_true",
                            help='Record lean records (only ego BEV and ego Lidar) for data logger')
        parser.add_argument('--hud', action="store_true",
                            help='Record the HUD display')
        parser.add_argument(
            '--eval', help='Autocast eval params (input: (ego_spd, ego_dist, col_spd, col_dist,[col_accel_dist]))',
            nargs='+',
            type=int)
        parser.add_argument('--sharing', action="store_true",
                            help='Enable sensor sharing ')
        # parser.add_argument('--commlog', action="store_true",
        #                     help='Enable schedule logging')
        parser.add_argument('--profile', action="store_true",
                            help='Enable pipeline latency profile')
        parser.add_argument('--timeout', default="10000.0",
                            help='Set the CARLA client timeout value in seconds')
        parser.add_argument('--bgtraffic', type=int, default=None,
                            help='Set the amount of background traffic in scenario')
        parser.add_argument('--bgtraffic_initspd', action="store_true",
                            help='Set a initial speed for background traffic')
        parser.add_argument('--nocollider', action="store_true",
                            help='Disable colliders in scenarios')
        parser.add_argument('--passive_collider', action="store_true",
                            help='Disable colliders sharing in scenarios')
        parser.add_argument('--outputdir', default='./result/',
                            help='Set the recordings output directory')
        parser.add_argument('--mqttport', type=int, default=1883,
                            help='Set the MQTT broker port (which should be different than port)')
        parser.add_argument('--noextrap', action="store_true",
                            help='Disable Extrapolation')
        parser.add_argument('--nofair', action="store_true",
                            help='Disable Fariness Compensation')
        parser.add_argument('--agnostic', action="store_true",
                            help='Disable Scheduling')
        parser.add_argument('--voronoi', action="store_true",
                            help='Running agnostic Voronoi baseline')
        parser.add_argument('--emulate', action="store_true",
                            help='Flag to switch between emulation (slower, higher fidelity, e.g. for communication, object detection) vs simulation (fast, use carla labels)')
        parser.add_argument('--fullpc', action="store_true",
                            help='Enable Full Point cloud sharing with 1000X bandwidth')
        parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + str(VERSION))

        parser.add_argument('--num_checkpoint', type=int, default=100,
                            help='Evaluating which checkpoint in the saved dir')
        parser.add_argument('--beta', type=float, default=0.9,
                            help='DAgger-Specific param, controls the sampling of expert vs. student action')
        parser.add_argument('--seed', type=int, default=0,
                            help='random seed used for background traffic')
        return parser


    @staticmethod
    def parse_config_flags(arguments):
        if arguments.emulate:
            Utils.EmulationMode = True

        if arguments.fullpc:
            Utils.object_oriented_sharing = False
            Utils.Rate = 1000 * Utils.Rate

        if arguments.noextrap:
            Utils.Extrapolation = False

        if arguments.agnostic:
            Utils.AGNOSTIC = True
            Utils.Extrapolation = False

        if arguments.voronoi:
            Utils.AGNOSTIC = True
            Utils.Extrapolation = False
            Utils.VORONOI = True
            Utils.Rate = Utils.V2XRate

        if arguments.outputdir:
            Utils.RecordingOutput = arguments.outputdir

        if arguments.full:
            Utils.DATALOG = True
            Utils.HUDLOG = True
            Utils.PATH_REUSE = False  # disable path reuse if data collection
        elif arguments.daggerdatalog:
            Utils.DATALOG = True
            Utils.HUDLOG = True
            Utils.DAGGERDATALOG = True
            Utils.PATH_REUSE = False
        elif arguments.lean:
            Utils.DATALOG = True
            Utils.HUDLOG = True
            Utils.LEANDATALOG = True
            Utils.PATH_REUSE = False  # disable path reuse if data collection
        elif arguments.hud:
            Utils.DATALOG = False
            Utils.HUDLOG = True
        else:
            Utils.DATALOG = False
            Utils.HUDLOG = False

        if arguments.profile:
            Utils.TIMEPROFILE = True

        if arguments.route:
            arguments.reloadWorld = True

        if arguments.commlog:
            Utils.COMMLOG = True

        if arguments.bgtraffic_initspd:
            Utils.BGTRAFFIC_INITSPD = True

        if arguments.nocollider:
            Utils.NO_COLLIDER = True

        if arguments.passive_collider:
            Utils.PASSIVE_COLLIDER = True
            print("Collider is NOT sharing")


    @staticmethod
    def transmission_time_sec(points_size, rate):
        return float(points_size*3*8*8/rate/1000000)
    
    ### this function is used to partition the view
    ### [input] points_dumb: point clouds from all views
    ###	   new_yaw: the yaw of the vehicle (for offset)
    ###	   index: the index of the partitioned view
    ###	   V: the total number of views
    ### [output] points_dumb: the point clouds in the specific view
    
    @staticmethod
    def view_partition(points_dumb, new_yaw, index, V):
        if points_dumb is None:
            return None
        single_angle = 360/V
        points_dumb_new = []
        points_angle = np.angle(points_dumb[...,0]+points_dumb[...,1]*1j, deg=True)+np.ones(shape=[points_dumb.shape[0],])*new_yaw
        points_angle = np.mod(points_angle,360)
        points_dumb = points_dumb[(single_angle*(index-1) <= points_angle) & (points_angle <= single_angle*index)]
        #print('Size of dumb data per view', points_dumb.shape)
        return points_dumb
    
    @staticmethod
    def get_dynamic_object(transmitter_id, point_clouds, nearby_objects):
        # print("========================================================")
        point_dumb = []
        objectlist = []
        TXTrans = CarlaActorPool.get_hero_actor().get_transform()
        for d in nearby_objects:
            if d is None:
                continue
            if d is not None:
                d_id = d.id
                if d_id == transmitter_id:
                    continue
    
            RXTrans = d.get_transform()
            point_clouds_temp = Utils.transform_pointcloud(point_clouds, TXTrans, RXTrans)
            point_clouds_temp = Utils.pc_to_car_alignment(point_clouds_temp)
    
            # print(point_clouds_temp.shape)
            peer_x_range = d.bounding_box.extent.x
            peer_y_range = d.bounding_box.extent.y
            peer_z_range = d.bounding_box.extent.z
            scaler = 1
            corners_x_max = peer_x_range / 2 + scaler
            corners_x_min = -peer_x_range / 2 - scaler
            corners_y_max = peer_y_range / 2 + scaler
            corners_y_min = -peer_y_range / 2 - scaler
            corners_z_max = peer_z_range / 2 + scaler
            corners_z_min = -peer_z_range / 2 - scaler
            # print(corners_z_min, corners_z_max)
            # point_clouds_temp = point_clouds_temp + [0, 0, LidarHeight]
            # point_clouds_x = point_clouds_temp[...,0]
            # point_clouds_y = point_clouds_temp[...,1]
            # point_clouds_z = point_clouds_temp[...,2]
    
            objectflag = False
            for p_index, p in enumerate(point_clouds_temp.tolist()):
                if (corners_x_min <= p[0]) and (p[0] <= corners_x_max) \
                        and (corners_y_min <= p[1]) and (p[1] <= corners_y_max) \
                        and (-scaler <= p[2]) and (p[2] <= peer_z_range+scaler):
                    point_dumb.append(p_index)
                    objectflag = True
            if objectflag:
                objectlist.append(d.id)
            #print(transmitter_id, d_index, len(point_dumb))
        return [point_clouds[point_dumb], objectlist]
    
    # def get_dynamic_object_specific(s, point_clouds, d):
    #     # print("========================================================")
    #     point_dumb = []
    #     TXTrans = s.get_transform()
    #     RXTrans = d.get_transform()
    #     point_clouds_temp = transform_pointcloud(point_clouds, TXTrans, RXTrans)
    #     point_clouds_temp = pc_to_car_alignment(point_clouds_temp)
    #
    #     # print(point_clouds_temp.shape)
    #     peer_x_range = d.bounding_box.extent.x
    #     peer_y_range = d.bounding_box.extent.y
    #     peer_z_range = d.bounding_box.extent.z
    #     scaler = 1
    #     corners_x_max = peer_x_range / 2 + scaler
    #     corners_x_min = -peer_x_range / 2 - scaler
    #     corners_y_max = peer_y_range / 2 + scaler
    #     corners_y_min = -peer_y_range / 2 - scaler
    #     corners_z_max = peer_z_range / 2 + scaler
    #     corners_z_min = -peer_z_range / 2 - scaler
    #     # print(corners_z_min, corners_z_max)
    #     point_clouds_temp = np.asarray(point_clouds_temp)
    #     # point_clouds_x = point_clouds_temp[...,0]
    #     # point_clouds_y = point_clouds_temp[...,1]
    #     # point_clouds_z = point_clouds_temp[...,2]
    #     point_clouds_temp2 = point_clouds_temp[(corners_x_min <= point_clouds_temp[...,0]) &
    #                                           (point_clouds_temp[...,0] <= corners_x_max) &
    #                                           (corners_y_min <= point_clouds_temp[...,1]) &
    #                                           (point_clouds_temp[...,1] <= corners_y_max)]
    #     #print(transmitter_id, d_index, len(point_dumb))
    #     return point_clouds_temp2
    
    @staticmethod
    def reachable_check(myTrans, peerTrans, r):
        peer_x = peerTrans.location.x
        peer_y = peerTrans.location.y
        my_x = myTrans.location.x
        my_y = myTrans.location.y
        distance = [peer_x - my_x, peer_y - my_y]
        if (distance[0]**2+distance[1]**2) <= r**2:
            return 1
        else:
            return 0

    @staticmethod
    def summarize_detected_object(detected_object_list):
        actorId_points_dict = dict()
        for obj in detected_object_list:
            if obj.actor_id not in actorId_points_dict:
                actorId_points_dict[obj.actor_id] = 0
            actorId_points_dict[obj.actor_id] += len(obj.point_cloud_list)
            # obj.print()
        return actorId_points_dict

    @staticmethod
    def find_weather_presets():
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]
    
    @staticmethod
    def get_actor_display_name(actor, truncate=250):
        name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
        return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name
    
    @staticmethod
    def inverse_transform(t):
        return carla.Transform(carla.Location(x=-t.location.x, y=-t.location.y, z=-t.location.z),
                               carla.Rotation(pitch=-t.rotation.pitch, yaw=-t.rotation.yaw, roll=-t.rotation.roll))
    
    
    
    # in yaw pitch roll sequence: from C++ source
    @staticmethod
    class TransformMatrix_WorldCoords(object):
        def __init__(self, transform):
            rotation = transform.rotation
            translation = transform.location
            cy = math.cos(np.radians(rotation.yaw))
            sy = math.sin(np.radians(rotation.yaw))
            cr = math.cos(np.radians(rotation.roll))
            sr = math.sin(np.radians(rotation.roll))
            cp = math.cos(np.radians(rotation.pitch))
            sp = math.sin(np.radians(rotation.pitch))
            self.matrix = np.matrix(np.identity(4))
            self.matrix[0, 3] = translation.x
            self.matrix[1, 3] = translation.y
            self.matrix[2, 3] = translation.z
            self.matrix[0, 0] = (cp * cy)
            self.matrix[0, 1] = (cy * sp * sr - sy * cr)
            self.matrix[0, 2] = -(cy * sp * cr + sy * sr)
            self.matrix[1, 0] = (sy * cp)
            self.matrix[1, 1] = (sy * sp * sr + cy * cr)
            self.matrix[1, 2] = (cy * sr - sy * sp * cr)
            self.matrix[2, 0] = (sp)
            self.matrix[2, 1] = -(cp * sr)
            self.matrix[2, 2] = (cp * cr)
            self.rcw = self.matrix[:3,:3]
            self.tcw= self.matrix[:3, 3]


        def inversematrix(self):
            """Return the inverse transform."""
            return np.linalg.inv(self.matrix)
    
    @staticmethod
    def pc_to_car_alignment(_pc):
        # Car alignment is a weird coordinate, upside down
        # Rest assured this is the right matrix, double checked
        alignment = np.array([[0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, -1]])
        return np.matmul(_pc, alignment)
    
    @staticmethod
    def pc_to_pic_alignment(_pc):
        alignment = np.array([[0, 1, 0],
                                [1, 0, 0],
                                [0, 0, -1]])
        return np.matmul(_pc, alignment)

    @staticmethod
    def car_to_pc_alignment(_pc):
        # Car alignment is a weird coordinate, upside down
        alignment = np.array([[0, 1, 0],
                                [-1, 0, 0],
                                [0, 0, -1]])
        pc_ret = np.matmul(_pc, np.linalg.inv(alignment))
        return pc_ret
    
    @staticmethod
    def car_to_world_transform(_pc, myTrans):
        alignment = np.array([[0, 1, 0],
                              [-1, 0, 0],
                              [0, 0, -1]])
        pc1 = np.matmul(_pc, alignment)
        pc = pc1.transpose()
        pc = np.append(pc, np.ones((1, pc.shape[1])), axis=0)
    
        mytrans = Utils.TransformMatrix_WorldCoords(myTrans)
    
        pc_transformed = mytrans.matrix * pc
        pc_rect = pc_transformed[0:3].transpose()
        return pc_rect
    
    @staticmethod
    def world_to_car_transform(_pc, myTrans):
        alignment = np.array([[0, 1, 0],
                              [-1, 0, 0],
                              [0, 0, -1]])
        pc = _pc.transpose()
        pc = np.append(pc, np.ones((1, pc.shape[1])), axis=0)
    
        mytrans = Utils.TransformMatrix_WorldCoords(myTrans)
    
        pc_transformed = mytrans.inversematrix() * pc
        pc_rect = pc_transformed[0:3].transpose()
        pc_return = np.array(np.matmul(pc_rect, np.linalg.inv(alignment)))
        return pc_return
    
    @staticmethod
    def transform_pointcloud(_pc, peerTrans, myTrans):
        # TODO: check all usage, compensate for lidar height difference, and forward placement
        peertrans = Utils.TransformMatrix_WorldCoords(peerTrans)
        mytrans = Utils.TransformMatrix_WorldCoords(myTrans)
        peertrans_mat = np.array(peertrans.matrix, dtype=np.float32)
        mytrans_inverse_mat = np.array(mytrans.inversematrix(),dtype=np.float32)

        alignment = np.array([[0.0, 1.0, 0.0],
                              [-1.0, 0.0, 0.0],
                              [0.0, 0.0, -1.0]], dtype=np.float32)
        _pc = np.asarray(_pc, dtype=np.float32)
        pc_ret = Utils.transform_pointcloud_jit(_pc,alignment, mytrans_inverse_mat, peertrans_mat)
        return pc_ret

    @staticmethod
    @jit(nopython=True)
    def transform_pointcloud_jit(_pc, alignment, mytrans_inverse_mat, peertrans_mat):
        """
        jit doesn't support np.matrix, but support np.array and np.dot
        """
        pc1 = np.dot(_pc, alignment)
        pc = pc1.transpose()
        pc = np.append(pc, np.ones((1, pc.shape[1]), dtype=np.float32), axis=0)
        # if trans are np.array
        pc_transformed = np.dot(peertrans_mat, pc)
        pc_transformed = np.dot(mytrans_inverse_mat, pc_transformed)
        # if trans are np.matrix
        # pc_transformed = mytrans_inverse_mat * (peertrans_mat * pc)
        pc_ret = pc_transformed[0:3].transpose()
        pc_ret = np.dot(pc_ret, np.linalg.inv(alignment))

        return pc_ret

    @staticmethod
    def transform_coords(_pc, peerTrans, myTrans):
        pc = _pc.transpose()
        pc = np.append(pc, np.ones((1, pc.shape[1])), axis=0)
    
        peertrans = Utils.TransformMatrix_WorldCoords(peerTrans)
        mytrans = Utils.TransformMatrix_WorldCoords(myTrans)
    
        pc_transformed = mytrans.inversematrix() * (peertrans.matrix * pc)
        pc_ret = pc_transformed[0:3].transpose()
        return pc_ret
    
    @staticmethod
    def map_to_robot_transform(_pc, robot_pose):
        pc = _pc.transpose()
        pc = np.append(pc, np.ones((1, pc.shape[1])), axis=0)
        robot_pose = Utils.TransformMatrix_WorldCoords(robot_pose)
        pc_robot_frame = np.matmul(robot_pose.inversematrix(), pc)
        pc_robot_frame = pc_robot_frame[0:3].transpose()
        pc_robot_frame = np.array(pc_robot_frame)
        return pc_robot_frame
    
    @staticmethod
    def robot_to_map_transform(_pc, robot_pose):
        pc = _pc.transpose()
        pc = np.append(pc, np.ones((1, pc.shape[1])), axis=0)
        robot_pose = Utils.TransformMatrix_WorldCoords(robot_pose)
        pc_map_frame = np.matmul(robot_pose.matrix, pc)
        pc_map_frame = pc_map_frame[0:3].transpose()
        pc_map_frame = np.array(pc_map_frame)
        return pc_map_frame

    @staticmethod
    def convert_json_to_transform(actor_dict):
        return carla.Transform(location=carla.Location(x=float(actor_dict["x"]),
                                                       y=float(actor_dict["y"]),
                                                       z=float(actor_dict["z"])),
                               rotation=carla.Rotation(roll=float(actor_dict["roll"]),
                                                       pitch=float(actor_dict["pitch"]),
                                                       yaw=float(actor_dict["yaw"]))
                               )
    
    @staticmethod
    def _lateral_shift(transform, shift):
        transform.rotation.yaw += 90
        transform.location += shift * transform.get_forward_vector()
        return transform
    
    @staticmethod
    def get_left_right_lane_marking(cur_waypoint):
        cur_lanewidth = cur_waypoint.lane_width
        cur_left_marking = carla.Transform(cur_waypoint.transform.location, cur_waypoint.transform.rotation)
        cur_right_marking = carla.Transform(cur_waypoint.transform.location, cur_waypoint.transform.rotation)
        cur_left_marking = Utils._lateral_shift(cur_left_marking, -cur_lanewidth * 0.5)
        cur_right_marking = Utils._lateral_shift(cur_right_marking, cur_lanewidth * 0.5)
        return cur_left_marking, cur_right_marking
    
    
    @staticmethod
    def perpendicular_distance(ego_transform, line_start_transform, line_end_transform):
        vec1 = np.array([ego_transform.location.x-line_start_transform.location.x,
                        ego_transform.location.y-line_start_transform.location.y,
                        ego_transform.location.z-line_start_transform.location.z])
        vec2 = np.array([ego_transform.location.x - line_end_transform.location.x,
                         ego_transform.location.y - line_end_transform.location.y,
                         ego_transform.location.z - line_end_transform.location.z])
        vec3 = np.array([line_start_transform.location.x - line_end_transform.location.x,
                         line_start_transform.location.y - line_end_transform.location.y,
                         line_start_transform.location.z - line_end_transform.location.z])
    
        d = norm(np.cross(vec1, vec2)) / norm(vec3)
        return d
    
    @staticmethod
    def calculate_lane_distance(cur_waypoint, target_waypoint, ego_transform):
        cur_left_marking, cur_right_marking = Utils.get_left_right_lane_marking(cur_waypoint)
        tar_left_marking, tar_right_marking = Utils.get_left_right_lane_marking(target_waypoint)
    
        left_distance = Utils.perpendicular_distance(ego_transform, cur_left_marking, tar_left_marking)
        right_distance = Utils.perpendicular_distance(ego_transform, cur_right_marking, tar_right_marking)
    
        return left_distance, right_distance
    
    @staticmethod
    def default_control():
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False
        return control
    
    @staticmethod
    def stop_control():
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        control.manual_gear_shift = False
        return control
    
    @staticmethod
    def speed_control(target_speed):
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 1.0*min(target_speed,20.0)/20.0
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False
        return control
    
    @staticmethod
    def get_distance(x1, x2, y1, y2):
        return math.sqrt((float(x1)-float(x2))**2 + (float(y1)-float(y2))**2)
    
    @staticmethod
    def get_geometric_linear_intersection_by_loc_and_intersection(ego_actor_loc, other_actor_loc):
        """
        Obtain a intersection point between two actor's location by using their waypoints (wp)
    
        @return point of intersection of the two vehicles
        """
    
        wp_ego_1 = CarlaDataProvider.get_map().get_waypoint(ego_actor_loc)
        wp_ego_2 = wp_ego_1.next(2)[0]
    
        while not wp_ego_2.is_intersection:
            wp_ego_2 = wp_ego_2.next(2)[0]
    
        x_ego_1 = wp_ego_1.transform.location.x
        y_ego_1 = wp_ego_1.transform.location.y
        x_ego_2 = wp_ego_2.transform.location.x
        y_ego_2 = wp_ego_2.transform.location.y
    
        # print("get_geometric_linear_intersection_by_loc_and_intersection ego 1 {} {}".format(x_ego_1, y_ego_1))
        # print("get_geometric_linear_intersection_by_loc_and_intersection ego 2 {} {}".format(x_ego_2, y_ego_2))
        wp_other_1 = CarlaDataProvider.get_world().get_map().get_waypoint(other_actor_loc)
        wp_other_2 = wp_other_1.next(2)[0]
        while not wp_other_2.is_intersection:
            wp_other_2 = wp_other_2.next(2)[0]
    
        x_other_1 = wp_other_1.transform.location.x
        y_other_1 = wp_other_1.transform.location.y
        x_other_2 = wp_other_2.transform.location.x
        y_other_2 = wp_other_2.transform.location.y
        # print("get_geometric_linear_intersection_by_loc_and_intersection actor 1 {} {}".format(x_other_1, y_other_1))
        # print("get_geometric_linear_intersection_by_loc_and_intersection actor 2 {} {}".format(x_other_2, y_other_2))
    
        s = np.vstack([(x_ego_1, y_ego_1), (x_ego_2, y_ego_2), (x_other_1, y_other_1), (x_other_2, y_other_2)])
        h = np.hstack((s, np.ones((4, 1))))
        line1 = np.cross(h[0], h[1])
        line2 = np.cross(h[2], h[3])
        x, y, z = np.cross(line1, line2)
        if z == 0:
            return None
    
        intersection = carla.Location(x=x / z, y=y / z, z=0)
    
        return intersection
    
    @staticmethod
    def lidar_to_hud_image(points, dim, maxdim, separator_index=0):
        lidar_img_size = (dim[0], dim[1], 3)
        lidar_img = np.zeros(lidar_img_size)

        if points is None or len(points) == 0:
            return lidar_img


        lidar_data = np.array(points[:, :2])
        # print(lidar_data.shape)
        # print(min(points[:,2]))
        # print(max(points[:, 2]))
        if separator_index == 0:
            separator_index = lidar_data.shape[0]
    
        lidar_data = lidar_data * min(dim) / maxdim
        # translate to display center
        lidar_data += (0.5 * dim[0], 0.5 * dim[1])
        # filter out of display dimension points
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = lidar_data[lidar_data[:, 0] < dim[0]]
        lidar_data = lidar_data[lidar_data[:, 1] < dim[1]]
        lidar_data = lidar_data[lidar_data[:, 0] >= 0]
        lidar_data = lidar_data[lidar_data[:, 1] >= 0]
    
        lidar_data_self = np.reshape(lidar_data[0:separator_index], (-1, 2))
        lidar_data_other = np.reshape(lidar_data[separator_index:], (-1, 2))
    
        lidar_img[tuple(lidar_data_self.T)] = (255, 255, 255)
        lidar_img[tuple(lidar_data_other.T)] = (255, 0, 0)
        return lidar_img
    
    @staticmethod
    def evict_till_outside_eviction_range(ego_location, trajectory_points_timestamp, grid_range):
        # waypoint eviction
        # ego_location = self._vehicle.get_location()
        max_index = -1 # if nothing out of range, then use last one
        for i in range(len(trajectory_points_timestamp)):
            distance = math.hypot(ego_location.x - trajectory_points_timestamp[i][0],
                                  ego_location.y - trajectory_points_timestamp[i][1])
            if distance < grid_range:
                max_index = i
    
        # # clear close waypoints
        # for i in range(max_index):
        #     try:
        #         trajectory_points_timestamp.pop(0)
        #     except:
        #         print(trajectory_points_timestamp)
        #         raise
        return trajectory_points_timestamp[max_index]
    
    @staticmethod
    def lidar_obj_2_xyz_numpy(lidar_obj):
        """
        a function to cope with the changes of lidar's flipped z axis in 0.9.10
        """
        points_raw = np.frombuffer(lidar_obj.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points_raw)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = points[:, :3]
        points[:, 2] = -points[:, 2]
        return points

    @staticmethod
    def compile_video(result_dir, output_name, glob_format=None):
        if glob_format is None:
            glob_format = ["*.jpg", "*.png"]
        for f in glob_format:
            file_names = result_dir + "/" + f
            print("Compiling Video for {}".format(file_names))
            files = glob.glob(file_names)
            if len(files) == 0:
                continue
            # files.sort()
            # print(files)
            files.sort(key=lambda x: int(x.split('/')[-1].split('.')[-2].split('_')[-1]))
            # print(files)


            out = None
            for filename in tqdm(files):
                img = cv2.imread(filename)
                if out is None:
                    height, width, layers = img.shape
                    size = (width, height)
                    out = cv2.VideoWriter(result_dir + "/" + output_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
                else:
                    out.write(img)
            if out is not None:
                out.release()
            break


    class EvalEnv(object):

        trace_id = None
        ego_speed_kmph = None
        ego_distance = None
        collider_speed_kmph = None
        collider_distance = None
        collider_accel_distance = None

        @staticmethod
        def tostring():
            return "TraceID {}, Ego spd {},dist {}, collider spd {}, dist {}, accel {}".format(
                Utils.EvalEnv.get_trace_id(), Utils.EvalEnv.ego_speed_kmph, Utils.EvalEnv.ego_distance, Utils.EvalEnv.collider_speed_kmph, Utils.EvalEnv.collider_distance, Utils.EvalEnv.collider_accel_distance)

        @staticmethod
        def get_trace_id():
            if Utils.EvalEnv.trace_id is not None:
                return Utils.EvalEnv.trace_id
            """create HUD output dir"""
            if not os.path.exists(Utils.RecordingOutput):
                os.makedirs(Utils.RecordingOutput, exist_ok=True)
            Utils.EvalEnv.trace_id = 0
            while os.path.exists(Utils.RecordingOutput + str(Utils.EvalEnv.trace_id)):
                Utils.EvalEnv.trace_id += 1
            os.mkdir(Utils.RecordingOutput + str(Utils.EvalEnv.trace_id))
            os.mkdir(Utils.RecordingOutput + str(Utils.EvalEnv.trace_id) + Utils.DebugOutput)
            return Utils.EvalEnv.trace_id

        @staticmethod
        def parse_eval_args(args):
            # args = args_string.split(',')
            if len(args) < 4:
                return
            Utils.EvalEnv.ego_speed_kmph = args[0]
            Utils.EvalEnv.ego_distance = args[1]
            Utils.EvalEnv.collider_speed_kmph = args[2]
            Utils.EvalEnv.collider_distance = args[3]
            if len(args) >= 5:
                Utils.EvalEnv.collider_accel_distance = args[4]
            print(Utils.EvalEnv.tostring())
