import os
import json
import yaml
import shutil

import numpy as np
import copy
import math
import imageio
import h5py

import carla

from AVR import Utils
from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider


class DataLogger():

    @staticmethod
    def record_sensor_dataset(frame_number, ego_action, agent_wrapper, route_id, ego_vehicle_id, SaveLidarImage=False):
        JsonState, action_noise = DataLogger.compile_actor_state(agent_wrapper._agent, frame_number, agent_wrapper)
        sensor_data = agent_wrapper._agent.sensor_interface.get_data_obj()
        DataLogger.add_data_point(agent_wrapper, ego_action, action_noise, sensor_data, JsonState, str(route_id),
                                  str(frame_number), str(ego_vehicle_id), SaveLidarImage=SaveLidarImage)

    @staticmethod
    def compile_actor_state(agent_instance, frame_number, agent_wrapper=None):

        ego_vehicle = CarlaActorPool.get_hero_actor()
        sensor_data = agent_instance.sensor_interface.get_data_obj()
        ego_transform = CarlaActorPool.get_hero_actor().get_transform()

        # add the ego trajectory output
        # ego_collaborator = agent_wrapper.get_collaborator_for_hud(ego_vehicle.id)
        agent_trajectory_points_timestamp = agent_instance.agent_trajectory_points_timestamp
        next_target_location = agent_instance.next_target_location

        cur_waypoint_carla = None
        tar_waypoint_carla = None
        cur_waypoint = "None"
        target_waypoint = "None"
        blocking_state = "None"
        road_option = "None"
        if agent_instance._agent is not None:
            if agent_instance._agent._local_planner._current_waypoint is not None:
                cur_waypoint_carla = agent_instance._agent._local_planner._current_waypoint
                cur_waypoint = DataLogger.convert_transform_to_dict(cur_waypoint_carla.transform)

            if agent_instance._agent._local_planner.target_waypoint is not None:
                tar_waypoint_carla = agent_instance._agent._local_planner.target_waypoint
                target_waypoint = DataLogger.convert_transform_to_dict(tar_waypoint_carla.transform)

            road_option = str(agent_instance._agent._local_planner._target_road_option)
            blocking_state = str(agent_instance._agent._state)
        # print(road_option)

        ego_velocity = ego_vehicle.get_velocity()
        ego_velocity_dict = dict()
        ego_velocity_dict.update({'vx': ego_velocity.x, 'vy': ego_velocity.y, 'vz': ego_velocity.z})

        # calculate lane distance
        left_distance = -1
        right_distance = -1
        if cur_waypoint_carla is not None and tar_waypoint_carla is not None:
            left_distance, right_distance = Utils.calculate_lane_distance(cur_waypoint_carla, tar_waypoint_carla,
                                                                          ego_transform)

        other_actors = DataLogger.compile_other_actors_state(agent_instance, agent_wrapper)
        state = dict()
        state.update({
            'ego_vehicle_id': ego_vehicle.id,
            'ego_vehicle_transform': DataLogger.convert_transform_to_dict(ego_transform),
            'blocking_state': blocking_state,
            'current_waypoint': cur_waypoint,
            'target_waypoint': target_waypoint,
            'left_distance': left_distance,
            'right_distance': right_distance,
            'target_road_option': road_option,
            # 'ego_vehicle_can_speed': sensor_data["can_bus"][1].data["speed"],
            'ego_vehicle_can_speed': np.linalg.norm((ego_velocity.x, ego_velocity.y, ego_velocity.z)),
            'ego_velocity': ego_velocity_dict,
            # 'ego_vehicle_can_timestamp': sensor_data["can_bus"][0],
            'ego_vehicle_can_timestamp': str(frame_number),
            'other_actors': other_actors,
            'next_target_location': next_target_location,
            'ego_planned_trajectory': agent_trajectory_points_timestamp,
        })

        if hasattr(agent_instance, 'steer_noise'):
            action_noise = carla.VehicleControl(steer=agent_instance.steer_noise)
        else:
            action_noise = None

        return state, action_noise

    @staticmethod
    def compile_control_states(c):
        state = {}
        state.update({
            'Throttle': c.throttle,
            'Steer': c.steer,
            'Brake': c.brake,
            'Reverse': c.reverse,
            'Hand brake': c.hand_brake,
            'Manual': c.manual_gear_shift,
            'Gear': c.gear
        })
        return state

    @staticmethod
    def detected_obj_by_detector_to_detectee(detected_objects_by_detector):
        detected_objects_by_detectee = dict()
        for detector_id in detected_objects_by_detector:
            for actor_id in detected_objects_by_detector[detector_id]:
                if actor_id not in detected_objects_by_detectee:
                    detected_objects_by_detectee[actor_id] = dict()
                detected_objects_by_detectee[actor_id][detector_id] = detected_objects_by_detector[detector_id][
                    actor_id]
        return detected_objects_by_detectee

    @staticmethod
    def compile_other_actors_state(agent_instance, agent_wrapper=None):
        # print("shared summary")
        shared_detected_objects = Utils.summarize_detected_object(agent_instance._agent.filtered_object_list)
        single_detected_objects = dict()
        actor_list = CarlaActorPool.get_actors()

        other_single_detected_objects_by_detectee = dict()
        other_shared_detected_objects_by_detectee = dict()
        if agent_wrapper is not None:
            ego_id = CarlaActorPool.get_hero_actor().id
            ego_collaborator = agent_wrapper.get_collaborator_for_hud(ego_id)
            # print("single summary")
            single_detected_objects = Utils.summarize_detected_object(ego_collaborator.filtered_detected_object_list)

            # other actor detected objects
            if Utils.scalability_eval:
                other_single_detected_objects = dict()
                other_shared_detected_objects = dict()
                for _, actor in actor_list:
                    other_collaborator = agent_wrapper.get_collaborator_for_hud(actor.id)
                    if other_collaborator is not None:
                        other_single_detected_objects[actor.id] = Utils.summarize_detected_object(
                            other_collaborator.filtered_detected_object_list)
                        other_shared_detected_objects[actor.id] = Utils.summarize_detected_object(
                            other_collaborator.filtered_detected_object_list_in_shared_pc)

                other_single_detected_objects_by_detectee = DataLogger.detected_obj_by_detector_to_detectee(
                    other_single_detected_objects)
                other_shared_detected_objects_by_detectee = DataLogger.detected_obj_by_detector_to_detectee(
                    other_shared_detected_objects)
        # print(other_single_detected_objects_by_detectee)

        other_actors = dict()
        actor_list = CarlaActorPool.get_actors()
        for _, actor in actor_list:
            # print(actor.id)
            if actor is None:
                # print("None actor")
                continue
            if not actor.is_alive:
                # print("Actor Not Alive")
                continue

            v = actor.get_velocity()
            v_dict = dict()
            v_dict.update({'vx': v.x, 'vy': v.y, 'vz': v.z})
            spd = math.sqrt(v.x ** 2 + v.y ** 2)
            actor_info = dict()
            actor_info.update({
                'type': actor.type_id,
                'transform': DataLogger.convert_transform_to_dict(actor.get_transform()),
                'velocity': spd,
                'velocity_vec': v_dict
            })

            if "vehicle" in actor.type_id or "walker" in actor.type_id:
                actor_info.update({
                    'bounding_box': DataLogger.convert_bounding_box_to_dict(actor.bounding_box)
                })
                if "vehicle" in actor.type_id:
                    c = actor.get_control()
                    actor_info.update(DataLogger.compile_control_states(c))

            if shared_detected_objects is not None and actor.id in shared_detected_objects:
                actor_info.update({'detected_quantpoints_after_sharing': shared_detected_objects[actor.id]})
            else:
                actor_info.update({'detected_quantpoints_after_sharing': 0})
            if single_detected_objects is not None and actor.id in single_detected_objects:
                actor_info.update({'detected_quantpoints': single_detected_objects[actor.id]})
            else:
                actor_info.update({'detected_quantpoints': 0})

            if Utils.scalability_eval:
                if actor.id in other_single_detected_objects_by_detectee and actor.id in other_shared_detected_objects_by_detectee:
                    actor_info.update(
                        {'other_shared_detected_objects': other_shared_detected_objects_by_detectee[actor.id],
                         'other_single_detected_objects': other_single_detected_objects_by_detectee[actor.id]})

            other_actors.update({actor.id: actor_info})
        return other_actors

    @staticmethod
    def add_data_point(agent_wrapper, control, control_noise, sensor_data, state, episode_number, frame_number,
                       ego_vehicle_id_str, SaveLidarImage=False):
        episode_path = os.path.join(Utils.RecordingOutput, episode_number, 'episode_' + episode_number.zfill(5))
        if not os.path.exists(episode_path):
            os.mkdir(episode_path)
            DataLogger.write_config(episode_path)
        DataLogger.write_json_measurements(episode_path, frame_number, control, control_noise, state)
        if Utils.DATALOG:
            DataLogger.write_sensor_data(agent_wrapper, episode_path, frame_number, sensor_data,
                                         ego_vehicle_id_str=ego_vehicle_id_str,
                                         LidarImage=SaveLidarImage)

    @staticmethod
    def write_config(episode_path):
        with open(os.path.join(episode_path, 'config.json'), 'w') as f:
            json.dump(Utils.environment_config.__dict__, f, indent=2)

    @staticmethod
    def write_json_measurements(episode_path, data_point_id,
                                # measurements,
                                control, control_noise,
                                state, write=True):

        # jsonObj = MessageToDict(measurements)
        # jsonObj.update(state)
        if control_noise is None:
            control_noise = carla.VehicleControl()
            control_noise.steer = 0.0
            control_noise.brake = 0.0
            control_noise.throttle = 0.0

        jsonObj = state
        jsonObj.update({'steer': control.steer})
        jsonObj.update({'throttle': control.throttle})
        jsonObj.update({'brake': control.brake})
        jsonObj.update({'hand_brake': control.hand_brake})
        jsonObj.update({'reverse': control.reverse})
        jsonObj.update({'steer_noise': control_noise.steer})
        jsonObj.update({'throttle_noise': control_noise.throttle})
        jsonObj.update({'brake_noise': control_noise.brake})

        if write:
            path = os.path.join(episode_path, 'measurements')
            if not os.path.exists(path):
                os.mkdir(path)

            fo = open(os.path.join(episode_path, 'measurements', data_point_id.zfill(5) + '.json'), 'w')
            fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))
            fo.close()

        return jsonObj

    @staticmethod
    def save_lidar_obj(lidar, filename, format='npy'):
        filename_ext = filename + '.' + format
        if format == 'npy':
            points = Utils.lidar_obj_2_xyz_numpy(lidar)
            np.save(filename_ext, points)
        elif format == 'bin':
            points = Utils.lidar_obj_2_xyz_numpy(lidar)
            points.tofile(filename_ext)
        elif format == 'ply':
            lidar.save_to_disk(filename_ext)
        elif format == 'h5':
            points = Utils.lidar_obj_2_xyz_numpy(lidar)
            h5f = h5py.File(filename_ext, 'w')
            h5f.create_dataset(filename, data=points)
            h5f.close()

    @staticmethod
    def write_sensor_data(agent_wrapper,
                          episode_path,
                          data_point_id,
                          sensor_data,
                          no_lidar=False,
                          ego_vehicle_id_str=None,
                          LidarImage=False):

        for name, data in sensor_data.items():
            if Utils.LEANDATALOG:
                # record ego data only for lean data logs, use --full to enable peer vehicle data logging
                if ego_vehicle_id_str not in name:
                    continue

            if 'RGB' in name:
                format = '.png'
                if ego_vehicle_id_str in name and not Utils.DAGGERDATALOG:
                    data[1].save_to_disk(os.path.join(episode_path, name, data_point_id.zfill(5) + format))
            elif 'Left' in name:
                format = '.png'
                if ego_vehicle_id_str in name and not Utils.DAGGERDATALOG:
                    data[1].save_to_disk(os.path.join(episode_path, name, data_point_id.zfill(5) + format))
            elif 'Right' in name:
                format = '.png'
                if ego_vehicle_id_str in name and not Utils.DAGGERDATALOG:
                    data[1].save_to_disk(os.path.join(episode_path, name, data_point_id.zfill(5) + format))
            elif 'LIDAR' in name:
                if no_lidar:
                    continue
                points = Utils.lidar_obj_2_xyz_numpy(data[1])
                format = 'npy'
                lidar_path = os.path.join(episode_path, name)
                if not os.path.exists(lidar_path):
                    os.mkdir(lidar_path)
                filename = os.path.join(lidar_path, data_point_id.zfill(5))
                DataLogger.save_lidar_obj(data[1], filename, format=format)
                if LidarImage:
                    DataLogger.save_lidar_jpeg(points, lidar_path, data_point_id)


                '''Fused Lidar'''
                if not Utils.DAGGERDATALOG:
                    fused_lidar_path = os.path.join(episode_path, name + 'Fused')
                    if not os.path.exists(fused_lidar_path):
                        os.mkdir(fused_lidar_path)
                    index_lidar_path = os.path.join(episode_path, name + 'Index')
                    if not os.path.exists(index_lidar_path):
                        os.mkdir(index_lidar_path)

                    vehicle_id = int(name.split('_')[0])
                    # obtain fused lidar points
                    collaborator = agent_wrapper.get_collaborator_for_hud(vehicle_id)
                    if collaborator is not None:
                        fused_points = collaborator.fused_sensor_data
                        np.save(os.path.join(fused_lidar_path, data_point_id.zfill(5) + ".npy"), fused_points)
                        # save where the ego_lidar index ends
                        # label = collaborator.ego_lidar_ending_index
                        # np.save(os.path.join(index_lidar_path, data_point_id.zfill(5) + ".npy"), label)

                    if LidarImage and not Utils.DAGGERDATALOG:
                        DataLogger.save_fused_lidar_jpeg(agent_wrapper, vehicle_id, fused_lidar_path, data_point_id)

    @staticmethod
    def save_lidar_jpeg(points, lidar_path, data_point_id, separator_index=0):
        format = '.jpg'
        points = Utils.pc_to_car_alignment(points)
        lidar_img = Utils.lidar_to_hud_image(points, dim=[720, 720], maxdim=Utils.LidarRange * 2 + 1,
                                             separator_index=separator_index)
        lidar_img = lidar_img.astype(np.uint8)
        lidar_img = np.flipud(lidar_img)  # upsidedown, vertical flip
        imageio.imwrite(os.path.join(lidar_path, data_point_id.zfill(5) + format), lidar_img)

    @staticmethod
    def save_fused_lidar_jpeg(agent_wrapper, vehicle_id, lidar_path, data_point_id):
        collaborator = agent_wrapper.get_collaborator_for_hud(vehicle_id)
        if collaborator is None:
            print("collaborator is None")
            return
        points = collaborator.fused_sensor_data
        label = collaborator.ego_lidar_ending_index
        DataLogger.save_lidar_jpeg(points, lidar_path, data_point_id, separator_index=label)

    @staticmethod
    def make_dataset_path(dataset_path):
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

    @staticmethod
    def add_metadata(dataset_path, settings_module):
        with open(os.path.join(dataset_path, 'metadata.json'), 'w') as fo:
            jsonObj = {}
            jsonObj.update(settings_module.sensors_yaw)
            jsonObj.update({'fov': settings_module.FOV})
            jsonObj.update({'width': settings_module.WINDOW_WIDTH})
            jsonObj.update({'height': settings_module.WINDOW_HEIGHT})
            jsonObj.update({'lateral_noise_percentage': settings_module.lat_noise_percent})
            jsonObj.update({'longitudinal_noise_percentage': settings_module.long_noise_percent})
            jsonObj.update({'car range': settings_module.NumberOfVehicles})
            jsonObj.update({'pedestrian range': settings_module.NumberOfPedestrians})
            jsonObj.update({'set_of_weathers': settings_module.set_of_weathers})
            fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))

    @staticmethod
    def add_episode_metadata(dataset_path, episode_number, episode_aspects):

        if not os.path.exists(os.path.join(dataset_path, 'episode_' + episode_number)):
            os.mkdir(os.path.join(dataset_path, 'episode_' + episode_number))

        with open(os.path.join(dataset_path, 'episode_' + episode_number, 'metadata.json'), 'w') as fo:
            jsonObj = {}
            jsonObj.update({'number_of_pedestrian': episode_aspects['number_of_pedestrians']})
            jsonObj.update({'number_of_vehicles': episode_aspects['number_of_vehicles']})
            jsonObj.update({'seeds_pedestrians': episode_aspects['seeds_pedestrians']})
            jsonObj.update({'seeds_vehicles': episode_aspects['seeds_vehicles']})
            jsonObj.update({'weather': episode_aspects['weather']})
            fo.write(json.dumps(jsonObj, sort_keys=True, indent=4))

    # Delete an episode in the case
    @staticmethod
    def delete_episode(dataset_path, episode_number):
        shutil.rmtree(os.path.join(dataset_path, 'episode_' + episode_number))

    @staticmethod
    def convert_transform_to_dict(transform):
        trans = dict()
        trans.update({
            'x': transform.location.x,
            'y': transform.location.y,
            'z': transform.location.z,
            'roll': transform.rotation.roll,
            'pitch': transform.rotation.pitch,
            'yaw': transform.rotation.yaw,
        })
        return trans

    @staticmethod
    def convert_bounding_box_to_dict(box):
        box_dict = dict()
        box_dict.update({
            'loc_x': box.location.x,
            'loc_y': box.location.y,
            'loc_z': box.location.z,
            'extent_x': box.extent.x,
            'extent_y': box.extent.y,
            'extent_z': box.extent.z,
        })
        return box_dict
