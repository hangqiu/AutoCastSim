from AVR import Utils, Collaborator
from AVR.Utils import transform_pointcloud, convert_json_to_transform, transform_coords
import carla
import pygame
import scipy
from scipy import misc

import json
from collections import namedtuple

from AVR.PCProcess import LidarPreprocessor

import numpy as np
import os
import h5py
from PIL import Image, ImageDraw
import math

import imageio

lidar_dir = "LIDAR"
merged_lidar_img_dir = "MergedLIDAR_IMG"
metadata_img_dir = "Meta_IMG"
lidar_img_dir = "LIDAR_IMG"
rgb_dir = ["Left", "Right"]
meta_dir = "measurements"

ACTOR_NUM=10
PEER_DATA_SIZE=13
META_DATA_SIZE=13
META_DATA_SIZE_IN_MODEL=10
N_ACTION = 15

MEM_LENGTH = 5
GAP_FRAMES = 2

class Episode_Data:
    # version add actor_ego_distance and actor yaw, and ego
    """
    An episode data contains data in a frame snapshot:

    ego_meta
        cur_ego_loc.x,
        cur_ego_loc.y,
        cur_ego_loc.z,
        json_dict["ego_vehicle_can_speed"],
        json_dict["ego_vehicle_transform"]['yaw'] / 180.0,
        json_dict["left_distance"],
        json_dict["right_distance"],
        current_waypoint_rela_loc.x,
        current_waypoint_rela_loc.y,
        current_waypoint_rela_loc.z,
        target_waypoint_rela_loc.x,
        target_waypoint_rela_loc.y,
        target_waypoint_rela_loc.z

    peer_data (zero padding, indicated by peer_data_mask)
        actor_type,
        actor_spd_ego_perspective,
        actor_yaw_ego_perspective,
        actor_velocity_ego_perspective['vx'],
        actor_velocity_ego_perspective['vy'],
        actor_velocity_ego_perspective['vz'],
        actor_ego_dist,
        actor_loc_ego_perspective[0],
        actor_loc_ego_perspective[1],
        actor_loc_ego_perspective[2],
        actor_extent[0],
        actor_extent[1],
        actor_extent[2],

    ego_lidar
    ego_lidar_depth
    merged_lidar_depth
    ego_rgb_left
    ego_rgb_right
    ego_actions

    """
    def __init__(self, episode_dir=None, last_frame_cache=None):
        if episode_dir is None:
            self.id = "tmp"
        else:
            self.id = episode_dir.split('/')[-1]
        self._episode_dir = episode_dir

        # misc
        self.ego_id = None
        self.timestamp = []
        self.ego_transform = []  # x, y, z, yaw, pitch, roll
        ### data to be recovered
        # lidar
        self.lidar_dim = LidarPreprocessor.lidar_dim
        self.lidar_depth_dim = LidarPreprocessor.lidar_depth_dim
        self.ego_lidar = np.zeros((0, self.lidar_dim[0], self.lidar_dim[1], self.lidar_dim[2]))
        self.ego_lidar_depth = np.zeros((0, self.lidar_depth_dim[0], self.lidar_depth_dim[1], 1))
        self.merged_lidar_depth = np.zeros((0, self.lidar_depth_dim[0], self.lidar_depth_dim[1], 1))
        """
            create a merged lidar memory of mem_length (default 5, which is 0.5 s) 
            of frame gaps (default of 2, FAF use 1 on 10fps, we have 20fps)
        """
        # self.mem_length = 5 # per 0.5 s
        # self.frame_gap = 2 # per 100 ms
        # self.merged_lidar_cache = []
        # self.merged_lidar_depth_mem = np.zeros((0, self.lidar_depth_dim[0], self.lidar_depth_dim[1], self.mem_depth))

        # rgb
        self.rgb_left = None #np.zeros((0, 200, 300, 3))
        self.rgb_right = None #np.zeros((0, 200, 300, 3))

        # ego metadata
        # self.ego_speed = np.zeros((0, 1, 1))
        self.ego_metadata_size = META_DATA_SIZE
        self.ego_meta = np.zeros((0, 1, self.ego_metadata_size))

        # actions
        self.n_action = 3  # throttle, brake, steer
        self.ego_actions = np.zeros((0, self.n_action))

        # intermediate results
        # """
        # Give a vector of 7 X 4 = 28 grid of 20m X 20m in the 140 X 80 BEV area
        # Each grid can have upto 5 actors, thats 28 X 5 = 140 actors... enough padding
        # Each actor would have 9 point,
        # Each actor has a id, type and velocity: 1: vehicle, 2: walker, 0:None
        # 6 of which, in ego perspective, are loc point, x,y,z and extent point x,y,z (the far boundary corner)
        # """
        # self.dX = 20
        # self.dY = 20
        # self.X_grid_num = 4
        # self.Y_grid_num = 7
        # self.grid_num = self.X_grid_num * self.Y_grid_num
        # self.grid_actor_num = 5

        """set a vector of 30 vehicle around"""

        self.actor_num = ACTOR_NUM

        self.peer_data_size = PEER_DATA_SIZE
        self.peer_data = np.zeros((0, self.actor_num, self.peer_data_size))
        self.peer_data_mask = np.zeros((0, 1, self.actor_num))

        self.last_frame_actor_cache = np.zeros(self.actor_num)
        if last_frame_cache is not None:
            self.last_frame_actor_cache = last_frame_cache

    def compile_peer_data_from_JSON(self, json_dict):
        """
            remove ID, keep speed, remove bounding box, add yaw, add distance between actor and ego

            peer meta vector format:

            actor_type,
            actor_spd_ego_perspective,
            actor_yaw_ego_perspective,
            actor_velocity_ego_perspective['vx'],
            actor_velocity_ego_perspective['vy'],
            actor_velocity_ego_perspective['vz'],
            actor_ego_dist,
            actor_loc_ego_perspective[0],
            actor_loc_ego_perspective[1],
            actor_loc_ego_perspective[2],
            actor_extent[0],
            actor_extent[1],
            actor_extent[2],
        """
        # init to all
        current_frame_actor_cache = np.zeros(self.actor_num)

        if self.ego_id is None:
            self.ego_id = json_dict["ego_vehicle_id"]

        peer_data = np.zeros((1, self.actor_num, self.peer_data_size))
        peer_data_mask = np.zeros((1, 1, self.actor_num))
        ego_carla_transform = convert_json_to_transform(json_dict["ego_vehicle_transform"])

        peer_count = 0
        # the first peer is ego itself
        """format: type, spd, yaw, velocity(x,y,z), distance, location(x,y,z), boundingbox_extent(x,y,z) (absolute)"""
        """all relatives except for ego vehicle"""
        ego_velocity = json_dict["ego_velocity"]
        ego_pd = np.array([1,
                           json_dict["ego_vehicle_can_speed"], ego_carla_transform.rotation.yaw / 180.0,
                           ego_velocity["vx"], ego_velocity["vy"], ego_velocity["vz"],
                           0, 0, 0, 0,
                           2.4543750286102295, 0.9279687404632568, 0.6399999856948853 # lincoln mkz
                        ])

        peer_data[0][peer_count] = ego_pd
        peer_data_mask[0][0][peer_count] = 1
        peer_count += 1
        current_frame_actor_cache[0] = int(self.ego_id)


        for actor_id in json_dict["other_actors"]:
            if actor_id == self.ego_id:
                continue
            type_id = json_dict["other_actors"][actor_id]["type"]
            if not ("vehicle" in type_id or "walker" in type_id):
                continue
            actor_transform_dict = json_dict["other_actors"][actor_id]["transform"]
            actor_carla_transform = convert_json_to_transform(actor_transform_dict)
            # distance from ego to actor
            actor_ego_dist = ego_carla_transform.location.distance(actor_carla_transform.location)

            # to account for lidar position height difference... hard-coded consistent with how sensor is generated
            # so that we can directly apply actor and ego transform to transform the lidar
            actor_bbox = json_dict["other_actors"][actor_id]["bounding_box"]
            actor_carla_transform.location.z -= (actor_bbox["loc_z"] + actor_bbox["extent_z"] + 0.1 - 1.60)

            actor_loc_ori = np.array([actor_bbox["loc_x"], actor_bbox["loc_y"], actor_bbox["loc_z"]])
            actor_extent = np.array([actor_bbox["extent_x"], actor_bbox["extent_y"], actor_bbox["extent_z"]])
            actor_extent_loc_ori = actor_extent + actor_loc_ori
            # print([actor_loc, actor_extent])
            [actor_loc_ego_perspective, actor_extent_loc_ego_perspective] = np.asarray(transform_coords(np.array([actor_loc_ori, actor_extent_loc_ori]),
                                                                    actor_carla_transform,
                                                                    ego_carla_transform))

            actor_loc_ego_perspective = Utils.car_to_pc_alignment(actor_loc_ego_perspective)
            actor_extent_loc_ego_perspective = Utils.car_to_pc_alignment(actor_extent_loc_ego_perspective)

            if not LidarPreprocessor.is_in_BEV(actor_loc_ego_perspective[0], actor_loc_ego_perspective[1], actor_loc_ego_perspective[2]):
                # print("Actor {} (loc:{}) not in BEV".format(actor_id, actor_loc))
                continue
            # print([actor_loc, actor_extent])
            actor_spd = json_dict["other_actors"][actor_id]["velocity"]
            actor_velocity = json_dict["other_actors"][actor_id]["velocity_vec"]
            actor_velocity_ego_perspective = dict()
            actor_velocity_ego_perspective.update({
                'vx': actor_velocity['vx']-ego_velocity['vx'],
                'vy': actor_velocity['vy']-ego_velocity['vy'],
                'vz': actor_velocity['vz']-ego_velocity['vz']
            })
            actor_spd_ego_perspective = math.sqrt(actor_velocity_ego_perspective['vx']**2 + actor_velocity_ego_perspective['vy']**2)

            actor_yaw_ego_perspective = (actor_carla_transform.rotation.yaw - ego_carla_transform.rotation.yaw) / 180.0

            actor_type = 0
            if "vehicle" in json_dict["other_actors"][actor_id]["type"]:
                actor_type = 1
            if "walker" in json_dict["other_actors"][actor_id]["type"]:
                actor_type = 2

            """format: type, spd, yaw, velocity(x,y,z), distance, location(x,y,z), boundingbox_extent(x,y,z) (absolute)"""
            """all relatives except for ego vehicle"""
            pd = np.array([actor_type,
                           actor_spd_ego_perspective, actor_yaw_ego_perspective,
                           actor_velocity_ego_perspective['vx'], actor_velocity_ego_perspective['vy'], actor_velocity_ego_perspective['vz'],
                           actor_ego_dist,
                           actor_loc_ego_perspective[0], actor_loc_ego_perspective[1], actor_loc_ego_perspective[2],
                           actor_extent[0], actor_extent[1], actor_extent[2],
                           ])
            # print(pd)
            # find ID position in the cache
            ID_position = 0
            for i in range (0,self.actor_num):
                if self.last_frame_actor_cache[i] == int(actor_id):
                    ID_position = i
                    break
                    
            if (ID_position == 0):
                for i in range (0,self.actor_num):
                    if self.last_frame_actor_cache[i] == 0:
                        ID_position = i
                        break
            if (ID_position == 0):
                continue
            
            current_frame_actor_cache[ID_position] = int(actor_id) 

            peer_data[0][ID_position] = pd
            peer_data_mask[0][0][ID_position] = 1
            peer_count += 1

            if peer_count >= self.actor_num:
                break

            self.last_frame_actor_cache = current_frame_actor_cache
        return peer_data, peer_data_mask

    def parse_LidarFile_from_JSON(self, json_dict, BEV_Depth, frame_id):
        # iterate thru all peers, and convert lidar to ego
        BEV_Depth_Merged_max = BEV_Depth
        ego_carla_transform = convert_json_to_transform(json_dict["ego_vehicle_transform"])
        # peer_count = 0
        for actor_id in json_dict["other_actors"]:
            if actor_id == self.ego_id:
                continue
            if not ("vehicle" in json_dict["other_actors"][actor_id]["type"]):
                # print("Not a Vehicle")
                continue
            actor_transform_dict = json_dict["other_actors"][actor_id]["transform"]
            actor_carla_transform = convert_json_to_transform(actor_transform_dict)
            # to account for lidar position height difference... hard-coded consistent with how sensor is generated
            # so that we can directly apply actor and ego transform to transform the lidar
            actor_bbox = json_dict["other_actors"][actor_id]["bounding_box"]
            actor_carla_transform.location.z -= (actor_bbox["loc_z"] + actor_bbox["extent_z"] + 0.1 - 1.60)

            # Lidar data
            actor_lidar_fp = os.path.join(self._episode_dir, actor_id + Collaborator.LidarSensorName, frame_id + ".npy")
            if os.path.exists(actor_lidar_fp):
                actor_lidar_data = np.load(actor_lidar_fp)
                actor_lidar_data_ego_perspective = transform_pointcloud(actor_lidar_data,
                                                                        actor_carla_transform,
                                                                        ego_carla_transform)
                # merged_lidar = np.concatenate([merged_lidar, actor_lidar_data_ego_perspective], axis=0)

                [_, actor_BEV_Depth_max, _] = LidarPreprocessor.Lidar2BEV(actor_lidar_data_ego_perspective)
                actor_BEV_Depth_max = np.reshape(actor_BEV_Depth_max, (actor_BEV_Depth_max.shape[0], actor_BEV_Depth_max.shape[1], 1))
                BEV_Depth_Merged_max = np.maximum(BEV_Depth_Merged_max, actor_BEV_Depth_max)
        return BEV_Depth_Merged_max

    def compile_meta_frame_from_JSON(self, json_dict):
        """
        egp meta vector format:

        cur_ego_loc.x,
        cur_ego_loc.y,
        cur_ego_loc.z,
        json_dict["ego_vehicle_can_speed"],
        json_dict["ego_vehicle_transform"]['yaw'] / 180.0,
        json_dict["left_distance"],
        json_dict["right_distance"],
        current_waypoint_rela_loc.x,
        current_waypoint_rela_loc.y,
        current_waypoint_rela_loc.z,
        target_waypoint_rela_loc.x,
        target_waypoint_rela_loc.y,
        target_waypoint_rela_loc.z
        """
        current_waypoint = json_dict["current_waypoint"]
        target_waypoint = json_dict["target_waypoint"]
        cur_ego_pos = json_dict["ego_vehicle_transform"]

        current_waypoint_rela_loc = carla.Location(x=0, y=0, z=0)
        target_waypoint_rela_loc = carla.Location(x=0, y=0, z=0)
        cur_ego_loc = carla.Location(x=0, y=0, z=0)

        if (current_waypoint != "None") and (target_waypoint != "None") and (cur_ego_pos != "None"):
            # print(current_waypoint)
            # print(target_waypoint)
            # print(cur_ego_transform)
            current_waypoint_trans = convert_json_to_transform(current_waypoint)
            target_waypoint_trans = convert_json_to_transform(target_waypoint)
            cur_ego_trans = convert_json_to_transform(cur_ego_pos)

            current_waypoint_rela_loc = current_waypoint_trans.location - cur_ego_trans.location
            target_waypoint_rela_loc = target_waypoint_trans.location - cur_ego_trans.location

            cur_ego_loc = carla.Location(x=cur_ego_trans.location.x, y=cur_ego_trans.location.y,
                                         z=cur_ego_trans.location.z)

        meta_frame = np.array([cur_ego_loc.x, cur_ego_loc.y, cur_ego_loc.z,
                               json_dict["ego_vehicle_can_speed"], json_dict["ego_vehicle_transform"]['yaw'] / 180.0,
                               json_dict["left_distance"], json_dict["right_distance"],
                               current_waypoint_rela_loc.x, current_waypoint_rela_loc.y, current_waypoint_rela_loc.z,
                               target_waypoint_rela_loc.x, target_waypoint_rela_loc.y, target_waypoint_rela_loc.z])

        meta_frame = np.reshape(meta_frame, (1, 1, self.ego_metadata_size))
        return meta_frame

    def append(self, metadata_file):
        frame_id = metadata_file.split('.')[0]
        meta_fp = os.path.join(self._episode_dir, meta_dir, metadata_file)
        meta_f = open(meta_fp)
        json_dict = json.load(meta_f)

        # misc
        if self.ego_id is None:
            self.ego_id = json_dict["ego_vehicle_id"]

        self.timestamp.append(json_dict["ego_vehicle_can_timestamp"])
        self.ego_transform.append(json_dict["ego_vehicle_transform"])

        # lidar
        ego_lidar_data_dir = os.path.join(self._episode_dir, "{}_{}".format(self.ego_id, lidar_dir), frame_id + ".npy")
        if os.path.exists(ego_lidar_data_dir):
            ego_lidar_data = np.load(ego_lidar_data_dir)

            merged_lidar = ego_lidar_data

            [BEV, BEV_Depth_Max, _] = LidarPreprocessor.Lidar2BEV(ego_lidar_data)
            BEV = LidarPreprocessor.occupancy_grid_dict_to_numpy(BEV)
            BEV = np.reshape(BEV, (1, BEV.shape[0], BEV.shape[1], BEV.shape[2]))
            self.ego_lidar = np.concatenate([self.ego_lidar, BEV], axis=0)
            BEV_Depth_Max = np.reshape(BEV_Depth_Max, (1, BEV_Depth_Max.shape[0], BEV_Depth_Max.shape[1], 1))
            self.ego_lidar_depth = np.concatenate([self.ego_lidar_depth, BEV_Depth_Max], axis=0)

            BEV_Depth_Merged = self.parse_LidarFile_from_JSON(json_dict, BEV_Depth_Max, frame_id)
            # peer_data = self.compile_peer_data_from_JSON(json_dict)
            peer_data, peer_data_mask = self.compile_peer_data_from_JSON(json_dict)

            self.merged_lidar_depth = np.concatenate([self.merged_lidar_depth, BEV_Depth_Merged], axis=0)
            # self.peer_data = np.concatenate([self.peer_data, peer_data], axis=0)
            self.peer_data = np.concatenate([self.peer_data, peer_data], axis=0)
            self.peer_data_mask = np.concatenate([self.peer_data_mask, peer_data_mask], axis=0)
            # print(self.peer_data_yaw_distance)


        # rgb
        left_rgb_dir = os.path.join(self._episode_dir, "{}_{}".format(self.ego_id,rgb_dir[0]), frame_id + ".png")
        if os.path.exists(left_rgb_dir):
            left_rgb = imageio.imread(left_rgb_dir)[:, :, :3]
            left_rgb = np.reshape(left_rgb, (1, left_rgb.shape[0], left_rgb.shape[1], left_rgb.shape[2]))
            if self.rgb_left is None:
                self.rgb_left = left_rgb
            else:
                self.rgb_left = np.concatenate([self.rgb_left, left_rgb], axis=0)
        # right rgb
        right_rgb_dir = os.path.join(self._episode_dir, "{}_{}".format(self.ego_id,rgb_dir[1]), frame_id + ".png")
        if os.path.exists(right_rgb_dir):
            right_rgb = imageio.imread(right_rgb_dir)[:, :, :3]
            right_rgb = np.reshape(right_rgb, (1, right_rgb.shape[0], right_rgb.shape[1], right_rgb.shape[2]))
            if self.rgb_right is None:
                self.rgb_right = right_rgb
            else:
                self.rgb_right = np.concatenate([self.rgb_right, right_rgb], axis=0)

        # meta data
        meta_frame = self.compile_meta_frame_from_JSON(json_dict)

        self.ego_meta = np.concatenate([self.ego_meta, meta_frame], axis=0)
        # print('spd_yaw: {}'.format(self.ego_speed_yaw))
        # road option
        # road_option = np.array(json_dict["target_road_option"])
        # road_option = np.reshape(road_option, (1, 1, 1))
        # self.road_option = np.concatenate([self.road_option, spd], axis=0)

        # actions
        ego_actions = np.array([[json_dict["throttle"],
                                 json_dict["steer"],
                                 json_dict["brake"]]])
        self.ego_actions = np.concatenate([self.ego_actions, ego_actions], axis=0)
        # # lidar memory
        # self.merged_lidar_cache.append(merged_lidar)
        # while len(self.merged_lidar_cache) > self.mem_length:
        #     self.merged_lidar_cache.pop(0)
        #
        # merged_lidar_depth_mem = BEV_Depth_Merged
        # for i in range(self.mem_length):
        #     # from end to start
        #     if i >= len(self.merged_lidar_cache):
        #         merged_lidar_depth_mem = n
        #
        #     index = -(i+1)
        #     tmp_lidar = self.merged_lidar_cache[index]
        #     tmp_ego_trans = self.ego_transform[index]
        #     tmp_lidar_ego_perspective = transform_pointcloud(tmp_lidar,
        #                                                             tmp_ego_trans,
        #                                                             self.ego_transform[-1])
        #     _, tmp_lidar_BEV_Depth = self.lidar_preprocessor.Lidar2BEV(tmp_lidar_ego_perspective)
        return self.last_frame_actor_cache


class DataParser:
    # version add yaw, distance
    def __init__(self, data_dir, display):
        self._data_dir = data_dir
        episode_dir = os.listdir(self._data_dir)
        self._episodes = []
        '''
        for ed in episode_dir:
            if ed[-3:-1] == "txt":
                continue
            self._episodes.append(os.path.join(ed, 'episode_'+ed.zfill(5)))
        '''
        for ed in episode_dir:
            if ed.startswith("episode"):
                self._episodes.append(ed)
        print("_episodes", self._episodes)
        self._episodes_data = []
        self._display = display

    def parse_episode(self, episode, model_name="PIXOR", batch=5):
        """
        parse the data folder, for each episode, generate data for training
        :return: input: [RGB_left, RGB_right, Lidar, LidarMerged, speed(speed,yaw), other_actors(yaw,distance)]
        """
        episode_data = Episode_Data(os.path.join(self._data_dir, episode))
        dir = os.path.join(self._data_dir, episode, meta_dir)
        count = 0


        lidar_file_tmp = episode + "_tmp.h5"
        lidar_file = episode + ".h5"
        lidar_fp = os.path.join(self._data_dir, episode, lidar_file)
        lidar_fp_tmp = os.path.join(self._data_dir, episode, lidar_file_tmp)
        if os.path.exists(lidar_fp_tmp):
            os.remove(lidar_fp_tmp)

        last_frame_actor_cache = np.zeros((ACTOR_NUM))
        
        for frame in sorted(os.listdir(dir)):
            print("frame : ",frame)
            last_frame_actor_cache = episode_data.append(frame)
            # print("test cache {}".format(last_frame_actor_cache))

            #debug
            # print("ego meta: {}".format(episode_data.ego_meta))
            # print("ego action: {}".format(episode_data.ego_actions))
            # print("peer data: {}".format(episode_data.peer_data))
            # print("peer data mask: {}".format(episode_data.peer_data_mask))

            if count != 0 and count % batch == 0:
                if not os.path.exists(lidar_fp_tmp):
                    hf = h5py.File(lidar_fp_tmp, 'w')
                    if model_name=="PIXOR":
                        hf.create_dataset('ego_lidar_depth', data=episode_data.ego_lidar_depth,
                                          maxshape=(None, episode_data.ego_lidar_depth.shape[1], episode_data.ego_lidar_depth.shape[2], episode_data.ego_lidar_depth.shape[3]))
                        hf.create_dataset('merged_lidar_depth', data=episode_data.merged_lidar_depth,
                                          maxshape=(None, episode_data.merged_lidar_depth.shape[1],episode_data.merged_lidar_depth.shape[2],episode_data.merged_lidar_depth.shape[3]))
                    # hf.create_dataset('ego_speed', data=episode_data.ego_speed,
                    #                   maxshape=(None, episode_data.ego_speed.shape[1], episode_data.ego_speed.shape[2]))
                    # Add yaw to speed
                    hf.create_dataset('ego_meta', data=episode_data.ego_meta,
                                      maxshape=(None, episode_data.ego_meta.shape[1], episode_data.ego_meta.shape[2]))
                    hf.create_dataset('ego_actions', data=episode_data.ego_actions,
                                      maxshape=(None, episode_data.ego_actions.shape[1]))
                    # hf.create_dataset('rgb_left', data=episode_data.rgb_left,
                    #                   maxshape=(None, episode_data.rgb_left.shape[1], episode_data.rgb_left.shape[2], episode_data.rgb_left.shape[3]))
                    # hf.create_dataset('rgb_right', data=episode_data.rgb_right,
                    #                   maxshape=(None, episode_data.rgb_right.shape[1], episode_data.rgb_right.shape[2],episode_data.rgb_right.shape[3]))
                    hf.create_dataset('peer_data', data=episode_data.peer_data,
                                      maxshape=(None, episode_data.peer_data.shape[1], episode_data.peer_data.shape[2]))
                    hf.create_dataset('peer_data_mask', data=episode_data.peer_data_mask,
                                      maxshape=(None, episode_data.peer_data_mask.shape[1], episode_data.peer_data_mask.shape[2]))
                    hf.close()
                else:
                    hf = h5py.File(lidar_fp_tmp, 'a')
                    if model_name == "PIXOR":
                        hf["ego_lidar_depth"].resize((hf["ego_lidar_depth"].shape[0] + episode_data.ego_lidar_depth.shape[0]), axis=0)
                        hf["ego_lidar_depth"][-episode_data.ego_lidar_depth.shape[0]:] = episode_data.ego_lidar_depth
                        hf["merged_lidar_depth"].resize((hf["merged_lidar_depth"].shape[0] + episode_data.merged_lidar_depth.shape[0]), axis=0)
                        hf["merged_lidar_depth"][-episode_data.merged_lidar_depth.shape[0]:] = episode_data.merged_lidar_depth
                    hf["ego_meta"].resize((hf["ego_meta"].shape[0] + episode_data.ego_meta.shape[0]), axis=0)
                    hf["ego_meta"][-episode_data.ego_meta.shape[0]:] = episode_data.ego_meta
                    hf["ego_actions"].resize((hf["ego_actions"].shape[0] + episode_data.ego_actions.shape[0]), axis=0)
                    hf["ego_actions"][-episode_data.ego_actions.shape[0]:] = episode_data.ego_actions
                    # hf["rgb_left"].resize((hf["rgb_left"].shape[0] + episode_data.rgb_left.shape[0]), axis=0)
                    # hf["rgb_left"][-episode_data.rgb_left.shape[0]:] = episode_data.rgb_left
                    # hf["rgb_right"].resize((hf["rgb_right"].shape[0] + episode_data.rgb_right.shape[0]), axis=0)
                    # hf["rgb_right"][-episode_data.rgb_right.shape[0]:] = episode_data.rgb_right
                    hf["peer_data"].resize((hf["peer_data"].shape[0] + episode_data.peer_data.shape[0]), axis=0)
                    hf["peer_data"][-episode_data.peer_data.shape[0]:] = episode_data.peer_data
                    hf["peer_data_mask"].resize((hf["peer_data_mask"].shape[0] + episode_data.peer_data_mask.shape[0]), axis=0)
                    hf["peer_data_mask"][-episode_data.peer_data.shape[0]:] = episode_data.peer_data_mask
                    hf.close()


                del episode_data
                episode_data = Episode_Data(os.path.join(self._data_dir, episode), last_frame_actor_cache)




            ## debug BEV
            # save_lidar_BEV_depth_as_img(episode_data.ego_lidar_depth[-1], self._data_dir, episode, count, save_dir=lidar_img_dir)
            # save_lidar_BEV_depth_as_img(episode_data.merged_lidar_depth[-1], self._data_dir, episode, count, save_dir=merged_lidar_img_dir)

            ## debug peer metadata
            # save_meta_BEV_as_img(episode_data.peer_data[-1], self._data_dir, episode, count, save_dir=metadata_img_dir)

            count += 1

        os.rename(lidar_fp_tmp, lidar_fp)

def save_lidar_BEV_depth_as_img(lidar_bev_depth, data_dir, episode, count, save_dir):
    normalized_BEV_depth = lidar_bev_depth + 10
    normalized_BEV_depth = normalized_BEV_depth * 100 + 50
    normalized_BEV_depth = np.reshape(normalized_BEV_depth,
                                      (normalized_BEV_depth.shape[0], normalized_BEV_depth.shape[1]))
    # print(normalized_BEV_depth.shape)
    # self._surface = pygame.surfarray.make_surface(normalized_BEV_depth.astype(int))
    # self._display.blit(self._surface, (0, 0))

    format = '.jpg'
    normalized_BEV_depth = np.transpose(normalized_BEV_depth)
    debug_dir = os.path.join(data_dir, episode, save_dir)
    if not os.path.exists(debug_dir):
        os.mkdir(debug_dir)
    scipy.misc.imsave(os.path.join(data_dir, episode, save_dir, str(count).zfill(5) + format),
                      normalized_BEV_depth)


def save_meta_BEV_as_img(peer_data, data_dir, episode, count, save_dir):
    im = Image.new("RGB", (800, 1400))
    dr = ImageDraw.Draw(im)
    for i in range(peer_data.shape[0]):
        for j in range(peer_data.shape[1]):
            if peer_data[i][j][0] == -1:
                continue
            loc_x = (peer_data[i][j][3] + 40)*10
            loc_y = (peer_data[i][j][4] + 70)*10
            extent_x = (peer_data[i][j][6] + 40)*10
            extent_y = (peer_data[i][j][7] + 70)*10
            dr.rectangle((loc_x, loc_y, extent_x, extent_y), outline=(255,255,255))
    format = '.jpg'
    debug_dir = os.path.join(data_dir, episode, save_dir)
    if not os.path.exists(debug_dir):
        os.mkdir(debug_dir)
    im.save(os.path.join(data_dir, episode, save_dir, str(count).zfill(5) + format))
