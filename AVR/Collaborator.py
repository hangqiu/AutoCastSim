import random
import carla
import copy
import weakref
import numpy as np
import threading
import time
import queue
import imageio
import os
import open3d as o3d


from AVR import PCProcess, Sched, Utils, Comm
from AVR.DataLogger import DataLogger
from AVR.DetectedObject import DetectedObject
from AVR.ViewSegment import ViewSegment
from AVR.CommLogger import CommLogger
from AVR.PCProcess import LidarPreprocessor
from AVR.TraceLogger import TraceLogger
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool

LidarSensorName = "_LIDAR"
FusedLidarSensorName = "_FusedLidar"
ProcessedLidarSensorName = "_ProcessedLidar"

class VehicleState(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.FrameID = -1
        # self.InScheduleSession = False
        # self.clusterHeadId = None
        self.myTrans = None
        self.speed = None
        self.PeerList_DataChannel = []
        self.PeerList_ControlChannel = []

    def get_state(self):
        # TODO: these are mutables. use deepcopy
        self.lock.acquire()
        state = VehicleState()
        state.FrameID = self.FrameID
        # state.InScheduleSession= self.InScheduleSession
        # state.clusterHeadId = self.clusterHeadId
        state.myTrans = self.myTrans
        state.speed = self.speed
        state.PeerList_DataChannel = self.PeerList_DataChannel
        state.PeerList_ControlChannel = self.PeerList_ControlChannel
        self.lock.release()
        return state

    def set_state(self, _fid, _my_transform, _speed, _control_peers, _data_peers):
        self.lock.acquire()
        self.FrameID = _fid
        self.myTrans = _my_transform
        self.speed = _speed
        self.PeerList_DataChannel = _data_peers
        self.PeerList_ControlChannel = _control_peers
        self.lock.release()


class Collaborator(object):
    """
    Collaborator, under agent wrapper, has agent as member, knows all sensor interface
    """
    def __init__(self, _vehicle, _agent, _agent_wrapper, sharing_mode=False):

        if Utils.AgentDebug:
            self.debug_dir = "{}/{}/Debug/".format(Utils.RecordingOutput, str(Utils.EvalEnv.get_trace_id()))
            if not os.path.exists(self.debug_dir):
                os.mkdir(self.debug_dir)
        self.vehicle = _vehicle
        self.agent = _agent
        self.agent_wrapper = _agent_wrapper
        self.id = self.vehicle.id
        self.carlaworld = CarlaDataProvider.get_world()
        self.state = VehicleState()
        self.startFrameID = -1
        self.waypoints = []
        self.predict_trajectory = []
        self.last_waypoints = []
        self.ProcessThread = None

        """Comm"""
        self.sharing_mode = sharing_mode
        self.ControlChannel = Comm.Comm(self, Comm.beacon_topic)
        self.DataChannel = Comm.Comm(self, Comm.data_topic)
        self.update_vehicle_state(-1)

        mState = self.state.get_state()
        myTrans = mState.myTrans
        boundingbox = self.vehicle.bounding_box
        b = Comm.Beacon(self.id, mState.FrameID,
                        myTrans.location.x, myTrans.location.y, myTrans.location.z,
                        myTrans.rotation.yaw, myTrans.rotation.pitch, myTrans.rotation.roll,
                        boundingbox.extent.x, boundingbox.extent.y, boundingbox.extent.z,
                        mState.speed, [], {}, {}, False)
        self.ControlChannel.buffer[str(self.id)] = Comm.LockBuffer(b)
        # for carrier sensing simulation
        self.DataChannelTokenQueue = queue.Queue()

        """ Ring buffer"""
        self.cached_lidar_data = None
        self.cached_state = None


        self.filtered_detected_object_list = []
        self.filtered_detected_object_list_in_shared_pc = []
        self.drawing_object_list = []
        self.sensor_object = dict()
        self.sensor_object_new = dict()
        self.sensor_view = dict()
        self.id_object = dict()

        self.latest_lidar_data = None
        self.ego_lidar_ending_index = 0
        self.fused_sensor_data = np.empty(shape=[0, 3])
        self.LidarProcResult = dict() # dummy struct to pass sensor ready check
        self.fused_sensor_data_lock = threading.Lock()
        self.transformedPC = dict()

        """register an abstractive sensor: FusedLidar """
        self.fused_lidar_id = str(self.id) + FusedLidarSensorName
        self.agent.sensor_interface.register_sensor(self.fused_lidar_id, None)
        self.agent.sensor_interface.update_sensor(self.fused_lidar_id,
                                                  data=PCProcess.PointCloudFrame(-1, self.fused_sensor_data,
                                                                                 carla.Transform(), -1), data_obj=None,
                                                  timestamp=-1)

        """register an abstractive sensor: ProcessedLidar """
        self.proc_lidar_id = str(self.id) + ProcessedLidarSensorName
        self.agent.sensor_interface.register_sensor(self.proc_lidar_id, None)
        self.agent.sensor_interface.update_sensor(self.proc_lidar_id, data=self.LidarProcResult, data_obj=None,
                                                  timestamp=-1)

        """ Scheduler """
        self.scheduler = Sched.Sched(self.ControlChannel)
        self.untransmitted_peer_id = []
        self.untransmitted_obj_actor_id = []
        self.untransmitted_obj_reward = []

        """Logger"""
        self.logger = CommLogger(str(self.id))
        TraceLogger.try_register_vehicle(self.id)

        self.callback_lock = threading.Lock() # ensure there is only one call back running at a time
        self.weak_self = weakref.ref(self)
        # TODO: find the sensor from dataprovider
        # self.camera.listen(lambda image: Collaborator.callback(self.weak_self, image))

    def destroy(self):
        self.agent.sensor_interface.destroy_sensor(self.fused_lidar_id)
        self.agent.sensor_interface.destroy_sensor(self.proc_lidar_id)
        self.ControlChannel.destroy()
        self.DataChannel.destroy()
        print("waiting for collaborator to join")
        if self.ProcessThread is not None:
            self.ProcessThread.join()
        print("Destroyed Collaborator")

    def update_vehicle_state(self, fid):
        self.state.set_state(fid, self.vehicle.get_transform(), self.vehicle.get_velocity(),
                             list(self.ControlChannel.buffer.keys()), list(self.DataChannel.buffer.keys()))
        return self.state.get_state()

    """process sensor and do collaboration"""
    def tick(self):
        sensor_id = str(self.id) + LidarSensorName
        lidar_obj = self.agent.sensor_interface.get_data_obj_by_id(sensor_id)
        self.ProcessThread = threading.Thread(target=self.run_callback, args=[lidar_obj])
        self.ProcessThread.start()

    def tick_join(self):
        if self.ProcessThread is not None:
            self.ProcessThread.join()

    def is_alive(self):
        return self.vehicle.is_alive

    def run_callback(self, lidar_obj):

        if lidar_obj is None:
            return
        if self.latest_lidar_data is None:
            # points = np.frombuffer(lidar_obj.raw_data, dtype=np.dtype('f4'))
            # # 0.9.10
            # self.latest_lidar_data = np.reshape(points, (int(points.shape[0] / 4), 4))
            # self.latest_lidar_data = self.latest_lidar_data[:,:3]
            self.latest_lidar_data = Utils.lidar_obj_2_xyz_numpy(lidar_obj)

            self.cached_lidar_data = self.latest_lidar_data
            mState = self.state.get_state()
            self.cached_state = mState
            return

        self.callback_lock.acquire()
        # init
        if Utils.AgentDebug:
            print("Collaborator {} starts...".format(self.id))
        self.ControlChannel.sess.end_Session()
        """ Starting New Session"""
        frameId = lidar_obj.frame_number
        if self.startFrameID == -1:
            self.startFrameID = frameId

        # self.logger.logTXData_FrameCount(frameId)
        # self.logger.logRXBeacon_FrameCount(frameId)
        # self.logger.logRXData_FrameCount(frameId)

        # points = np.frombuffer(lidar_obj.raw_data, dtype=np.dtype('f4'))
        # self.latest_lidar_data = np.reshape(points, (int(points.shape[0] / 4), 4))
        # self.latest_lidar_data = self.latest_lidar_data[:, :3]

        self.latest_lidar_data = Utils.lidar_obj_2_xyz_numpy(lidar_obj)

        mState = self.update_vehicle_state(frameId)
        # TODO: check this is consistent over different vehicle
        # TODO: all following functions should use this state

        if self.sharing_mode:
            if Utils.TIMEPROFILE: print("\tCollaborator {}".format(self.id))
            start = time.time()
            self.object_detection(self.latest_lidar_data, frameId)
            self.fill_TX_queue(self.latest_lidar_data)
            lidar_time = time.time()
            if Utils.TIMEPROFILE: print("\t\tObject Detection: {} sec".format(lidar_time-start))

            """ Beacon and Data """
            # log directly to my Comm Buffer, don't wait for it to receive
            self.TX_Beacon(mState)

            if str(self.id) not in self.DataChannel.buffer.keys():
                self.init_datachannel_buffer(mState)

            beaconlist = self.RX_beacon(mState)

            schedule_vector, reward_vector, length_vector = self.compute_schedule(beaconlist, frameId)

            if not Utils.AGNOSTIC:
                # Save the first one into the agentwrapper
                schedule_reference = self.agent_wrapper.update_schedule(mState.FrameID, schedule_vector, beaconlist)
                # consistency check
                for i, ref_schedule in enumerate(schedule_reference):
                    local_schedule = schedule_vector[i]
                    if local_schedule[0] != ref_schedule[0] or local_schedule[1] != ref_schedule[1]:
                        print("Inconsistent schedule")
                        print(schedule_reference)
                        print(ref_schedule)
                        break


            sche_time = time.time()
            if Utils.TIMEPROFILE: print("\t\tSchedule: {} sec".format(sche_time - lidar_time))

            """model pkt loss"""
            # last_tx_idx_in_schedule = 0
            # if frameId % 5 == 0:
            last_tx_idx_in_schedule = self.TX_Data(mState, schedule_vector, length_vector)

            tx_time = time.time()
            if Utils.TIMEPROFILE:print("\t\tTX: {} sec".format(tx_time - sche_time))

            # """ Fairness Compensation """
            # if Utils.Fairness_Compensation:
            #     self.fairness_compensation(last_tx_idx_in_schedule, beaconlist, schedule_vector, reward_vector)

        rx_time_0 = time.time()
        self.RX_Fuse_Data(mState, frameId)
        rx_time = time.time()
        if Utils.TIMEPROFILE: print("\t\tRX: {} sec".format(rx_time - rx_time_0))

        ### Cache
        self.cached_state = mState
        self.cached_lidar_data = self.latest_lidar_data

        mSess = self.ControlChannel.sess.get_SessionState()
        self.logger.logSessionState(mSess, mState.FrameID)

        self.callback_lock.release()

        if Utils.AgentDebug:
            print("Collaborator {} ends...".format(self.id))


    def init_datachannel_buffer(self, mState):
        myTrans = mState.myTrans
        dummyData = Comm.Data(self.id, mState.FrameID, np.empty(shape=[0, 3]).tolist(),
                              myTrans.location.x, myTrans.location.y, myTrans.location.z,
                              myTrans.rotation.yaw, myTrans.rotation.pitch, myTrans.rotation.roll, -1,
                              -1)  # Need to be a list, not ndarray for json
        self.DataChannel.buffer[str(self.id)] = Comm.LockBuffer(dummyData)


    # def fairness_compensation(self, last_tx_idx, beaconlist, schedule_vector, reward_vector):
    #     self.untransmitted_peer_id = []
    #     self.untransmitted_obj_actor_id = []
    #     self.untransmitted_obj_reward = []
    #
    #     # print("untransmitted objs: {}-{}=?".format(len(schedule_vector), last_tx_idx, ))
    #     while last_tx_idx < len(schedule_vector):
    #
    #         peerId = schedule_vector[last_tx_idx][0]
    #         objId = schedule_vector[last_tx_idx][1]
    #         for i, beacon in enumerate(beaconlist):
    #             # print(beacon.id, peerId)
    #             if int(beacon.id) == int(peerId):
    #                 if objId < len(beacon.detected_object_list): # TODO: fix racing condition, cuz schedule is not using beacon list, but control channel buffer
    #                     obj = beacon.detected_object_list[objId]
    #                     self.untransmitted_peer_id.append(peerId)
    #                     self.untransmitted_obj_actor_id.append(obj.actor_id)
    #                     self.untransmitted_obj_reward.append(reward_vector[last_tx_idx])
    #                 break
    #         last_tx_idx += 1
    #
    #     # print(self.untransmitted_peer_id, self.untransmitted_obj_actor_id, self.untransmitted_obj_reward)




    def RX_beacon(self, mState):
        beaconlist = []
        if Utils.DistributedMode:
            # distributed beaconlist
            beaconlist = self.get_distributed_beaconlist(mState)
        else:
            # centralized beaconlist
            beaconlist = self.get_centralized_beaconlist()
        return beaconlist

    def get_distributed_beaconlist(self, mState):
        beaconlist = []
        for peerId in self.ControlChannel.buffer:
            beacon = self.ControlChannel.buffer[peerId].get_content()
            # check if the beacon is updated or stale, no need for distance check, cuz comm will take care of it
            print(f"current frame:{mState.FrameID}, beacon frame{beacon.FrameId}")
            if mState.FrameID != beacon.FrameId:
                continue
            beaconlist.append(beacon)
        return beaconlist

    def get_centralized_beaconlist(self):
        return self.agent_wrapper.get_beacon_list()
        # no need to check distance again for centralized one, when computing schedule, it will be filtered against cluster center

    def get_points_for_voronoi_based_sharing(self, lidar_data, debug=False):
        """
        return list of ego voronoi region points
        """
        if debug: print("Debugging Voronoi")
        beaconlist = self.agent_wrapper.get_beacon_list()
        actor_loc_list = []
        ego_index = -1
        for i, beacon in enumerate(beaconlist):
            if beacon.id == self.id:
                ego_index = i
            actor_loc_list.append([beacon.x, beacon.y, 0])
        if len(actor_loc_list) < 2 or ego_index == -1:
            return []
        actor_loc_array = np.array(actor_loc_list)
        if debug: print(actor_loc_array)
        ego_transform = self.vehicle.get_transform()
        actor_loc_array = Utils.world_to_car_transform(actor_loc_array, ego_transform)
        actor_loc_array = actor_loc_array[:,:2]
        if debug: print(actor_loc_array)

        actor_loc_list = list(actor_loc_array)
        vs = ViewSegment()
        vor = vs.get_voronoi(actor_loc_list)
        segments = vs.get_segments(vor)
        ego_segments = segments[ego_index]
        if debug: print(f"Ego segments {ego_segments}")

        counts = len(ego_segments)
        z = np.zeros([counts, 1])

        ego_segments_np = np.array(ego_segments)
        ego_segments_np = np.concatenate([ego_segments_np, z], axis=1)
        if debug: print(ego_segments_np)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_data)
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "Z"
        vol.axis_max = 500
        vol.axis_min = -500
        vol.bounding_polygon =  o3d.utility.Vector3dVector(ego_segments_np)
        cropped_pcd = vol.crop_point_cloud(pcd)

        if debug:
            o3d.io.write_point_cloud("test_trace.ply", pcd)
            o3d.io.write_point_cloud("test_trace_crop.ply", cropped_pcd)

        ego_TX_points = np.asarray(cropped_pcd.points)
        ego_TX_points = ego_TX_points.tolist()
        return ego_TX_points

    def fill_TX_queue(self, lidar_data):
        # push back to the TX queue
        if Utils.object_oriented_sharing:
            if not Utils.VORONOI:
                for ObjectId in range(len(self.filtered_detected_object_list)):
                    self.sensor_object_new[str(ObjectId)] = copy.deepcopy(self.filtered_detected_object_list[ObjectId])
            else:
                ### for voronoi baseline comparison
                TX_points = self.get_points_for_voronoi_based_sharing(lidar_data)
                obj = DetectedObject(0)
                obj.insert_point_cloud(TX_points)
                self.sensor_object_new['0'] = obj

        else:# view based sharing
            ### Update my view list
            for ViewId in range(0, int(Utils.View)):
                self.sensor_view[str(ViewId)] = Utils.view_partition(lidar_data,
                                                                     self.cached_state.myTrans.rotation.yaw, ViewId + 1,
                                                                     Utils.View)

                self.sensor_object[str(ViewId)] = np.empty(shape=[0, 3])
                self.id_object[str(ViewId)] = []

                actor_list = list(CarlaActorPool.get_actor_dict().values())

                if Utils.COMMLOG:
                    [self.sensor_object[str(ViewId)], self.id_object[str(ViewId)]] = Utils.get_dynamic_object(
                        self.id, self.sensor_view[str(ViewId)],
                        actor_list)  # TODO: change get_dynamic_object function, use this list of actors directly, instead of using old Vehicle Object


    def object_detection(self, lidar_data, frameId):

        ego_transform = self.vehicle.get_transform()

        start = time.time()
        if not Utils.Fast_Lidar:
            self.LidarProcResult = LidarPreprocessor.process_lidar(lidar_data, ego_transform,
                                                                   z_threshold=self.vehicle.bounding_box.extent.z * 2,
                                                                   ego_actor=self.vehicle,
                                                                   ego_length=self.vehicle.bounding_box.extent.x * 2,
                                                                   ego_width=self.vehicle.bounding_box.extent.y * 2)
        else:
            self.LidarProcResult = LidarPreprocessor.process_lidar_fast(lidar_data, ego_transform,
                                                                        z_threshold=self.vehicle.bounding_box.extent.z * 2,
                                                                        ego_length=self.vehicle.bounding_box.extent.x * 2,
                                                                        ego_width=self.vehicle.bounding_box.extent.y * 2)

        lidar_time = time.time()
        if Utils.TIMEPROFILE: print("\t\t\tLidar Processing: {} sec".format(lidar_time - start))
        self.agent.sensor_interface.update_sensor(self.proc_lidar_id, data=self.LidarProcResult, data_obj=None,
                                                  timestamp=frameId)
        sensor_time = time.time()
        if Utils.TIMEPROFILE: print("\t\t\tSensor update: {} sec".format(sensor_time- lidar_time))

        self.filtered_detected_object_list = LidarPreprocessor.estimated_actor_and_speed_from_detected_object(
            self.LidarProcResult.detected_object_list,
            ego_actor=self.vehicle)

        filter_time = time.time()
        if Utils.TIMEPROFILE: print("\t\t\tObject estimation: {} sec".format(filter_time - sensor_time))

        if Utils.AgentDebug:
            outputdir = self.debug_dir + "/{}_ego_obstacle_grid_{}.png".format(self.id, frameId)
            LidarPreprocessor.save_binary_occupancy_grid(outputdir, self.LidarProcResult.obstacle_grid)
            outputdir = self.debug_dir + "/{}_ego_obstacle_grid_with_margin_{}.png".format(self.id, frameId)
            LidarPreprocessor.save_binary_occupancy_grid(outputdir,
                                                         self.LidarProcResult.obstacle_grid_with_margin_for_planning)
            outputdir = self.debug_dir + "/{}_ego_filtered_actor_grid_{}.png".format(self.id, frameId)
            LidarPreprocessor.save_binary_occupancy_grid(outputdir, self.LidarProcResult.filtered_actor_grid)



    def compute_schedule(self, beaconlist, frameId):
        """
        Distributed scheduler:
            - Cluster center should be a predefined location, here use ego car's location for now, TODO: change to predefined HDMap location
            - beaconlist: if using centralized beaconlist, it's consistent, be aware of inconsistency when using distributed beaconlist
        """

        hero_actor = CarlaActorPool.get_hero_actor()
        cluster_center_trans = CarlaDataProvider.get_transform(hero_actor)

        # filter beacon based on cluster center and distance
        filtered_beaconlist = []
        for i, beacon in enumerate(beaconlist):
            ego_trans = carla.Transform(carla.Location(x=beacon.x, y=beacon.y, z=beacon.z))
            # filter out beacons outside of the cluster: 2 * Utils.HalfRadioRange around cluster center
            InRange = Utils.reachable_check(cluster_center_trans, ego_trans, 2 * Utils.HalfRadioRange)
            if InRange:
                filtered_beaconlist.append(beacon)
                if Utils.COMMLOG:
                    self.logger.logRXBeacon(beacon, frameId)

        if self.sharing_mode:
            succ = self.ControlChannel.sess.try_SessionBegin(hero_actor.id, frameId,
                                                             list(self.ControlChannel.buffer.keys()))
            if succ:
                self.scheduler.updateSchedule(beaconlist, self.untransmitted_peer_id,
                                              self.untransmitted_obj_actor_id,
                                              self.untransmitted_obj_reward)


        # if (mSess.InScheduleSession):
        [schedule_vector, reward_vector, length_vector, totReward, dV_vector, PeerIdMap, dO_reward_map, dO_actorId_map] = self.scheduler.getSchedule()

        self.logger.log_schedule(frameId, schedule_vector, length_vector, reward_vector, totReward, dV_vector, PeerIdMap, dO_reward_map, dO_actorId_map)

        # self.logger.logSchedule(schedule_vector, frameId)
        # self.logger.logReward(reward_vector, frameId)
        # self.logger.logTotReward(totReward, frameId)
        # self.logger.logDV(dV_vector, PeerIdMap, frameId)

        if Utils.TRACELOG:
            TraceLogger.log_schedule(self.id, frameId, Utils.EvalEnv.get_trace_id(), schedule_vector, reward_vector,
                                     length_vector)

        return schedule_vector, reward_vector, length_vector

    def TX_Data(self, mState, schedule_vector, length_vector):

        if len(schedule_vector) == 0:
            return 0
        time_counter_ms = 0
        for s_id, (CarId, ViewId) in enumerate(schedule_vector):
            time_ms = 0
            if CarId == str(self.id):
                time_ms = self.TX_Detected_Object(mState, ViewId, time_counter_ms)
            else:
                length = length_vector[s_id]
                time_ms = Utils.transmission_time_sec(length, Utils.Rate) * 1000.0

            time_counter_ms += time_ms
            # TX until 100ms slot filled
            if time_counter_ms >= 100:
                # print("100ms Exceeded")
                break

        if Utils.AgentDebug:
            print("Car {} Finished TX".format(self.id))

        return s_id

    def TX_Beacon(self, mState):
        ### Construct the Beacon
        myTrans = mState.myTrans
        speed = mState.speed
        boundingbox = self.vehicle.bounding_box
        trigger = not self.ControlChannel.sess.IsInSession()
        if (mState.FrameID - self.startFrameID < 5):
            trigger = False
            # print("trigger is not started yet")
            #observe state for 10 frames

        ### [Po-Han: TO DO] object -> simplified object list

        # print(self.id, mState.FrameID,
        #                 myTrans.location.x, myTrans.location.y, myTrans.location.z,
        #                 myTrans.rotation.yaw, myTrans.rotation.pitch, myTrans.rotation.roll,
        #                 boundingbox.extent.x, boundingbox.extent.y, boundingbox.extent.z,
        #                 speed, self.filtered_detected_object_list, self.sensor_view, self.sensor_object, trigger)

        objlist_for_comm = []
        # print(self.sensor_object_new)
        for k in self.sensor_object_new:
            o = self.sensor_object_new[k]
            #TODO: this is only for object oriented sharing, check compatibility, remove view based sharing
            objlist_for_comm.append(o.get_obj_for_comm())
        b = Comm.Beacon(self.id, mState.FrameID,
                        myTrans.location.x, myTrans.location.y, myTrans.location.z,
                        myTrans.rotation.yaw, myTrans.rotation.pitch, myTrans.rotation.roll,
                        boundingbox.extent.x, boundingbox.extent.y, boundingbox.extent.z,
                        speed, objlist_for_comm, self.sensor_view, self.sensor_object, trigger)
        # TODO: double check extent x,y,z with length, width, height

        # log directly to my Comm Buffer, don't wait for it to receive
        if self.id not in self.ControlChannel.buffer.keys():
            self.ControlChannel.buffer[str(self.id)] = Comm.LockBuffer(b)

        # Distributed beacons
        self.ControlChannel.pub_msg(b)
        # Centralized stable beacons
        self.ControlChannel.pub_ToBeaconList(b)

        self.logger.logTXBeacon(b, mState.FrameID)




    def TX_Detected_Object(self, mState, ItemID, time_ms):

        ### Construct the Data
        myTrans = mState.myTrans
        if Utils.object_oriented_sharing:
            points_dumb_return = np.array(self.sensor_object_new[str(ItemID)].point_cloud_list)
        else:
            if Utils.SlicedView:
                points_dumb_return = self.sensor_view[str(ItemID)]

        if Utils.TRACELOG:
            TraceLogger.log_Detected_Object_as_txt(self.id, mState.FrameID, Utils.EvalEnv.get_trace_id(),
                                                   ItemID, points_dumb_return)
            TraceLogger.log_Detected_Object_as_npy(self.id, mState.FrameID, Utils.EvalEnv.get_trace_id(),
                                                   ItemID, points_dumb_return)


        points_size = points_dumb_return.shape[0]
        if Utils.AgentDebug:
            print(str(self.id) + " Transmitting: " + str(points_dumb_return.shape) + " of View " + str(ItemID)
                  + " takes " + str(Utils.transmission_time_sec(points_size, Utils.Rate)))

        d = Comm.Data(self.id, mState.FrameID, points_dumb_return.tolist(),
                      myTrans.location.x, myTrans.location.y, myTrans.location.z,
                      myTrans.rotation.yaw, myTrans.rotation.pitch, myTrans.rotation.roll,
                      int(ItemID), time_ms)  # Need to be a list, not ndarray for json

        if Utils.EmulationMode:
            self.DataChannel.pub_msg(d)
        else:
            self.pub_ToPeerDataBuffer(d)

        time_ms = Utils.transmission_time_sec(points_size, Utils.Rate) * 1000.0

        return time_ms


    def pub_ToPeerDataBuffer(self, d):
        for c in self.agent_wrapper._collaborator_dict:
            peer_collaborator = self.agent_wrapper._collaborator_dict[c]
            t = threading.Thread(
                target=self.pub_ToPeerDataBuffer_thread,
                args=[peer_collaborator, d])
            t.start()

    def pub_ToPeerDataBuffer_thread(self, peer_collaborator, data):
        peer_collaborator.DataChannel.saveMsgToPeerBuffer(data)

    def RX_Fuse_Data(self, mState, frameId):

        freshness_threshold = 1 # was 10 frames, minimum 1
        # rx_start = time.time()

        myTrans = mState.myTrans
        self.transformedPC = dict()
        ThreadPool = dict()
        dataPeers = mState.PeerList_DataChannel
        controlPeers = mState.PeerList_ControlChannel

        for peerId in dataPeers:
            if (peerId not in controlPeers):
                continue
            if peerId == self.id:
                continue

            trans_start = time.time()
            data = self.DataChannel.buffer[peerId].get_content()
            if abs(mState.FrameID - data.FrameId) > freshness_threshold:
                continue
            beacon = self.ControlChannel.buffer[peerId].get_content()
            player_data = self.DataChannel.get_ndarray_from_Data(data)
            player_Trans = self.ControlChannel.get_carla_Transform_from_Data(data)
            # get_point_time = time.time()
            # if Utils.TIMEPROFILE: print("\t\t\t\t\t\tGet points: {} s".format(get_point_time - trans_start))

            self.transformedPC[peerId] = np.empty(shape=[0, 3])
            if player_data.shape[0] != 0:
                """ Lidar Height Difference Compensation """
                player_Trans.location.z += (beacon.DimZ-self.vehicle.bounding_box.extent.z) * 2
                new_points = Utils.transform_pointcloud(player_data, player_Trans, myTrans)
                # trans_time = time.time()
                # if Utils.TIMEPROFILE: print("\t\t\t\t\t\tTrans points: {} s".format(trans_time - get_point_time))
                self.transformedPC[peerId] = np.concatenate([self.transformedPC[peerId], new_points])
                # concat_time = time.time()
                # if Utils.TIMEPROFILE: print("\t\t\t\t\t\tConcat: {} s".format(concat_time - trans_time))


        fused_points = self.latest_lidar_data
        fused_label = self.latest_lidar_data.shape[0]

        # fuse_time = time.time()
        # if Utils.TIMEPROFILE: print("\t\t\t\tFusion: {} s".format(fuse_time - rx_start))

        for peerId in self.transformedPC.keys():
            fused_points = np.concatenate([fused_points, self.transformedPC[peerId]])

        if Utils.LOGCOVERAGE:
            coverage_count = PCProcess.ComputeCoverage(fused_points)
            self.logger.logCoverage(coverage_count,frameId)

        self.fused_sensor_data_lock.acquire()
        self.fused_sensor_data = fused_points
        # print("Collaborator lidar shape {}".format(self.fused_sensor_data.shape))
        self.ego_lidar_ending_index = fused_label
        pcframe = PCProcess.PointCloudFrame(self.id, self.fused_sensor_data, myTrans, frameId)
        # pcframe_time = time.time()
        # if Utils.TIMEPROFILE: print("\t\t\t\tPC frame: {} s".format(pcframe_time - fuse_time))
        self.agent.sensor_interface.update_sensor(self.fused_lidar_id, data=pcframe, data_obj=None, timestamp=frameId)
        self.fused_sensor_data_lock.release()

        # update_time = time.time()
        # if Utils.TIMEPROFILE: print("\t\t\t\tUpdate Sensor: {} s".format(update_time - pcframe_time))
        """ optionally process fused lidar for each collaborator, reference the processing of the agent """
        if Utils.scalability_eval:
            if not Utils.Fast_Lidar:
                self.LidarProcResult_SharedView = LidarPreprocessor.process_lidar(pcframe.pc, pcframe.trans,
                                                                                  z_threshold=self.vehicle.bounding_box.extent.z * 2,
                                                                                  ego_actor=self.vehicle,
                                                                                  ego_length=self.vehicle.bounding_box.extent.x * 2,
                                                                                  ego_width=self.vehicle.bounding_box.extent.y * 2)
            else:
                self.LidarProcResult_SharedView = LidarPreprocessor.process_lidar_fast(pcframe.pc, pcframe.trans,
                                                                                       z_threshold=self.vehicle.bounding_box.extent.z * 2,
                                                                                       ego_length=self.vehicle.bounding_box.extent.x * 2,
                                                                                       ego_width=self.vehicle.bounding_box.extent.y * 2)

            self.filtered_detected_object_list_in_shared_pc = LidarPreprocessor.estimated_actor_and_speed_from_detected_object(
                self.LidarProcResult_SharedView.detected_object_list,
                ego_actor=self.vehicle)


        if Utils.AgentDebug:
            print("Car" + str(self.id) + "PC Fused")

