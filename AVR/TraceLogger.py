import os
import threading

import numpy as np
from AVR import Utils

class TraceLogger():
    """
    to be combined into datalogger
    clean writeup for radio exps
    static functional class
    """
    vehicle_list = []
    vehicle_id_map = dict()
    output_dir = "trace"
    register_lock = threading.Lock()

    @staticmethod
    def try_register_vehicle(vid):
        vehicle_id = str(vid)
        if vehicle_id not in TraceLogger.vehicle_list:
            TraceLogger.register_lock.acquire()
            TraceLogger.vehicle_id_map[vehicle_id] = chr(ord('A') + len(TraceLogger.vehicle_list))
            TraceLogger.vehicle_list.append(vehicle_id)
            TraceLogger.register_lock.release()

    @staticmethod
    def get_mapped_vid(vehicle_id):
        vid = str(vehicle_id)
        if vid in TraceLogger.vehicle_id_map:
            return TraceLogger.vehicle_id_map[vid]
        return None

    @staticmethod
    def try_register_frame(route_id, frame_id):
        TraceLogger.register_lock.acquire()
        folder_name = os.path.join(Utils.RecordingOutput, str(route_id), TraceLogger.output_dir, str(frame_id))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        TraceLogger.register_lock.release()

    @staticmethod
    def log_Detected_Object_as_npy(vehicle_id, frame_id, route_id, item_id, pointcloud_list):
        TraceLogger.try_register_vehicle(vehicle_id)
        TraceLogger.try_register_frame(route_id, frame_id)
        """ as binary """
        folder_name = os.path.join(Utils.RecordingOutput, str(route_id), TraceLogger.output_dir, str(frame_id))
        file_name = folder_name + '/' + TraceLogger.get_mapped_vid(vehicle_id) + str(item_id) + '.npy'
        pc_array = np.array(pointcloud_list, dtype=float)
        pc_array = np.around(pc_array, decimals=4)
        np.save(file_name, pc_array)

    @staticmethod
    def log_Detected_Object_as_txt(vehicle_id, frame_id, route_id, item_id, pointcloud_list):
        TraceLogger.try_register_vehicle(vehicle_id)
        TraceLogger.try_register_frame(route_id, frame_id)
        """ as txt """
        folder_name = os.path.join(Utils.RecordingOutput, str(route_id), TraceLogger.output_dir, str(frame_id))
        file_name = folder_name + '/' + TraceLogger.get_mapped_vid(vehicle_id)+str(item_id) + '.txt'
        with open(file_name, "w") as file:
            for point in pointcloud_list:
                point = np.around(point, decimals=4)
                file.write(str(point))
                file.write('\n')
            file.write('F')

    # @staticmethod
    # def log_Detected_Object_as_bin(vehicle_id, frame_id, route_id, item_id, pointcloud_list):
    #     TraceLogger.try_register_vehicle(vehicle_id)
    #     TraceLogger.try_register_frame(route_id, frame_id)
    #     """ as binary """
    #     folder_name = os.path.join(Utils.RecordingOutput, str(route_id), TraceLogger.output_dir, str(frame_id))
    #     file_name = folder_name + '/' + TraceLogger.get_mapped_vid(vehicle_id) + str(item_id) + '.bin'
    #     with open(file_name, "wb") as file:
    #         # pc_array = np.array(pointcloud_list, dtype=float)
    #         # np.save(file_name, pc_array)
    #         for point in pointcloud_list:
    #             point = np.around(point, decimals=4)
    #             for i in range(0, len(point)):
    #                 file.write(point[i].tobytes())
    #             file.write(np.array(0.0000).tobytes())
    #         if len(pointcloud_list) == 0:
    #             file.write(np.array(0.0000).tobytes())
    #         file.write(np.array(0.0000).tobytes())


    @staticmethod
    def log_schedule(vehicle_id, frame_id, route_id, schedule_vector, reward_vector, length_vector):
        # print("Logging Schedule")
        # print(schedule_vector, reward_vector, length_vector)
        TraceLogger.try_register_vehicle(vehicle_id)
        TraceLogger.try_register_frame(route_id, frame_id)

        folder_name = os.path.join(Utils.RecordingOutput, str(route_id), TraceLogger.output_dir, str(frame_id))
        file_name = folder_name + '/schedule_{}.txt'.format(TraceLogger.get_mapped_vid(vehicle_id))
        TraceLogger.log_vehicle_id_map(folder_name)
        with open(file_name, "w") as file:
            for pair_id, (vid, objid) in enumerate(schedule_vector):
                TraceLogger.try_register_vehicle(vid)
                file.write(TraceLogger.get_mapped_vid(vid))
                file.write(str(objid))
                file.write('.txt, ')
                file.write(str(reward_vector[pair_id]))
                file.write(', ')
                file.write(str(length_vector[pair_id]))
                file.write(', ')
                file.write(str(Utils.transmission_time_sec(length_vector[pair_id], Utils.Rate) * 1000.0))
                file.write('\n')

    @staticmethod
    def log_vehicle_id_map(folder_name):
        file_name = folder_name + '/vehicle_id_map.txt'
        if os.path.exists(file_name):
            return
        with open(file_name, "w") as file:
            for v in TraceLogger.vehicle_id_map:
                file.write("{}:{}\n".format(v, TraceLogger.get_mapped_vid(v)))


