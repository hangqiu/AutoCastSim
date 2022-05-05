# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
import carla
import re
import pygame
import math
import numpy as np
import random
from AVR import Utils
import threading
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool

class Sched(object):
    def __init__(self, _mControlChannel):
        # self.parent = _parent()
        self.mControlChannel = _mControlChannel
        # self.mDataChannel = _mDataChannel
        self.frame_count = 0 # TODO: change to timestamp when available in carla

        # TODO: remove states, schedule module should not maintain states

        self.scheduleLock = threading.Lock()
        self.schedule = []
        self.length_vector = []
        self.totReward = 0.0
        self.reward_schedule = []
        self.dV_vector = []
        self.PeerIdMap = {}
        self.dO_reward_map = {}
        self.dO_actorId_map = {}

    def updateSchedule(self, beaconlist, untransmitted_peedId, untransmitted_actorId, untransmitted_reward):
        ret = self.ComputeSchedule(beaconlist, untransmitted_peedId, untransmitted_actorId, untransmitted_reward)
        if ret is None:
            return
        [_schedule, _reward_schedule, _length_vector, _totReward, _dV_vector, _PeerIdMap, _dO_reward_map, _dO_actorId_map] = ret
        self.scheduleLock.acquire()
        self.schedule = _schedule
        self.length_vector = _length_vector
        self.totReward = _totReward
        self.reward_schedule = _reward_schedule
        self.dV_vector = _dV_vector
        self.PeerIdMap = _PeerIdMap
        self.dO_reward_map=_dO_reward_map
        self.dO_actorId_map=_dO_actorId_map
        self.scheduleLock.release()

    def getSchedule(self):
        self.scheduleLock.acquire()
        s = self.schedule
        r = self.reward_schedule
        l = self.length_vector
        t = self.totReward
        dV = self.dV_vector
        idmap = self.PeerIdMap
        dO_reward_map = self.dO_reward_map
        dO_actorId_map = self.dO_actorId_map
        self.scheduleLock.release()
        return [s,r,l,t,dV, idmap, dO_reward_map, dO_actorId_map]

    def ComputeSchedule(self, beaconlist, untransmitted_peedId, untransmitted_actorId, untransmitted_reward):
        """
        compute schedule,


        """
        HalfRadioRange = Utils.HalfRadioRange
        num_vehicles = len(beaconlist)


        PeerIds = []
        for beacon in beaconlist:
            PeerIds.append(str(beacon.id))
        if (len(PeerIds)<2):
            return None

        PeerIds_map = {}
        PeerIds_reverse_map = {}
        count = 0
        for PeerId in sorted(PeerIds):
            PeerIds_map[PeerId] = str(count)
            PeerIds_reverse_map[str(count)] = PeerId
            count += 1

        TXRX_visibility_vector = np.zeros(shape=[num_vehicles,num_vehicles,Utils.View])
        debug_vector = np.zeros(shape=[num_vehicles,Utils.View])
        # using hero actor as cluster head (RSU) for now
        RSU_location = CarlaActorPool.get_hero_actor().get_transform().location
        clusterHeadTrans = carla.Transform(carla.Location(x=RSU_location.x, y=RSU_location.y))


        # Reachability check done in Comm, all peers are in 2*HalfRadioRange
        # select all those with HalfRadioRange of cluster head for a sharing Session
        dO_reward_map = {}
        dO_size_map = {}
        dO_actorId_map = {}
        controlPeer = beaconlist

        for idx, dL_Id in enumerate(PeerIds):
            # check first peer in half range
            dL_Beacon = beaconlist[idx]
            dL_trans = self.mControlChannel.get_carla_Transform_from_Beacon(dL_Beacon)

            if not Utils.reachable_check(clusterHeadTrans, dL_trans, HalfRadioRange):
                continue

            for idy, dR_Id in enumerate(PeerIds):
                if dL_Id == dR_Id:
                    continue
                # check second peer in half range
                dR_Beacon = beaconlist[idy]
                dR_trans = self.mControlChannel.get_carla_Transform_from_Beacon(dR_Beacon)
                if not Utils.reachable_check(clusterHeadTrans, dR_trans, HalfRadioRange):
                    continue
                # print("===================================")
                # print(f"{dL_Id} -> {dR_Id}: # objs = {len(dR_Beacon.detected_object_list)}")
                # print("===================================")
                """"Agnostic settings """
                blockage_vector = {}
                for vid in range(Utils.View):
                    blockage_vector[vid+1] = 1
                object_interest_vector = np.ones(shape=[len(dR_Beacon.detected_object_list)])
                """AutoCast settings"""
                if not Utils.AGNOSTIC:
                    blockage_vector = blockage_information(dL_trans,dR_trans,dR_Beacon.DimX, dR_Beacon.DimY, controlPeer, Utils.View)
                    object_interest_vector = blockage_information_rect(dL_trans, dR_trans, dR_Beacon.DimX, dR_Beacon.DimY, dL_Beacon.detected_object_list, dR_Beacon.detected_object_list)

                if PeerIds_map[dR_Id] not in dO_reward_map:
                    dO_reward_map[PeerIds_map[dR_Id]] = object_interest_vector
                    dO_size_map[PeerIds_map[dR_Id]] = []
                    dO_actorId_map[PeerIds_map[dR_Id]] = []
                    if Utils.object_oriented_sharing:
                        for detected_object in dR_Beacon.detected_object_list:
                            # detected_object.print()
                            dO_size_map[PeerIds_map[dR_Id]].append(len(detected_object.point_cloud_list))
                            dO_actorId_map[PeerIds_map[dR_Id]].append(detected_object.actor_id)
                    else:
                        print(dR_Beacon.cached_sensor_view_size)
                        dO_size_map[PeerIds_map[dR_Id]] = dR_Beacon.cached_sensor_view_size
                    dO_size_map[PeerIds_map[dR_Id]] = np.array(dO_size_map[PeerIds_map[dR_Id]])
                else:
                    dO_reward_map[PeerIds_map[dR_Id]] += object_interest_vector
                    #TODO: shouldn't reach here
                # print(dO_reward_map[dR_Id])
                # print(dO_size_map[PeerIds_map[dR_Id]])

                dL_index = int(PeerIds_map[dL_Id])
                dR_index = int(PeerIds_map[dR_Id])

                for v in blockage_vector:
                    TXRX_visibility_vector[dL_index][dR_index][v-1] = blockage_vector[v]
                    debug_vector[dR_index][v-1] += blockage_vector[v]

        received_vector = np.ones(shape=[num_vehicles,num_vehicles,Utils.View])
        totReward = total_reward(TXRX_visibility_vector, received_vector, Utils.View, HalfRadioRange)

        # if Utils.Fairness_Compensation:
        #     # print("Compensation:")
        #     for i, peerId in enumerate(untransmitted_peedId):
        #         # print(peerId)
        #         if peerId in PeerIds_map.keys():
        #             if PeerIds_map[peerId] in dO_reward_map.keys():
        #                 for j, actor_id in enumerate(dO_actorId_map[PeerIds_map[peerId]]):
        #                     # print(j, actor_id)
        #                     if actor_id == untransmitted_actorId[i]:
        #                         dO_reward_map[PeerIds_map[peerId]][j] += untransmitted_reward[i] * 5
        #                         # print("Compensating {} {} with {}".format(peerId, j, untransmitted_reward[i] * 5))

        if Utils.object_oriented_sharing:
            [policy_vector, reward_vector] = mckp_greedy_scheduling_algorithm(dO_reward_map, dO_size_map)
            if Utils.AGNOSTIC:
                temp = list(zip(policy_vector, reward_vector))
                random.shuffle(temp)
                policy_vector, reward_vector = zip(*temp)
        # elif Utils.autocast_mode:
        #     received_vector = np.ones(shape=[num_vehicles,num_vehicles,Utils.View])
        #     [policy_vector, reward_vector] = greedy_scheduling_algorithm(list(PeerIds_map.values()), dV_vector,received_vector, Utils.View, HalfRadioRange, Utils.SlotPerSession)
        else:
            received_vector = np.ones(shape=[num_vehicles, num_vehicles, Utils.View])
            [policy_vector, reward_vector] = roundrobin_scheduling_algorithm(list(PeerIds_map.values()), TXRX_visibility_vector, received_vector, Utils.View, HalfRadioRange, Utils.SlotPerSession)
        # print("Total Reward: "+ str(totReward))
        # print("*************************************")
        # print(policy_vector)
        # print("*************************************")
        policy_vector_new = []
        for policy in policy_vector:
            policy_vector_new.append([PeerIds_reverse_map[str(policy[0])], policy[1]])
        policy_vector = policy_vector_new

        size_vector = []
        # print(dO_reward_map.keys())
        # print(dO_size_map.keys())

        for CarId, ObjID in policy_vector:
            if (PeerIds_map[str(CarId)] in dO_size_map) and (ObjID < len(dO_size_map[PeerIds_map[str(CarId)]])):
                size_vector.append(dO_size_map[PeerIds_map[str(CarId)]][ObjID])
            else:
                # print(f"CarID{CarId}, objid{ObjID} not exist")
                size_vector.append(0)

        return [policy_vector, reward_vector, size_vector, totReward, TXRX_visibility_vector.tolist(), PeerIds_map, dO_reward_map, dO_actorId_map]


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
def mckp_greedy_scheduling_algorithm(dO_reward_map, dO_size_map):
    car_object_index_map = {}
    index = 0
    all_reward = []
    all_size = []
    greedy_metric = []
    for carId in dO_reward_map:
        for objectId, object_reward in enumerate(dO_reward_map[carId]):
            # filter out 0s: cannot do this, would cuz length not match
            # if dO_size_map[carId][objectId] == 0.0:
            #     continue
            # print("[DEBUG] " + str(objectId) + " \t" + str(object_reward))
            if object_reward > 0:
                car_object_index_map[index] = [int(carId), objectId]
                all_reward.append(object_reward+0.0001*int(carId)+0.00001*int(objectId))
                all_size.append(dO_size_map[carId][objectId])
                if dO_size_map[carId][objectId] == 0.0:
                    greedy_metric.append(0.0)
                else:
                    greedy_metric.append(all_reward[-1] / all_size[-1])
                index += 1

    all_reward = np.array(all_reward)
    all_size = np.array(all_size)
    greedy_metric = np.array(greedy_metric)

    # greedy_metric = all_reward/all_size
    greedy_index = np.argsort(greedy_metric)[::-1]
    policy_vector = []
    reward_vector = []
    for id in greedy_index:
        policy_vector.append(car_object_index_map[id])
        reward_vector.append(all_reward[id])
    # transmission_sec
    # print(policy_vector)
    return [policy_vector, reward_vector]

def reward_greedy_scheduling_algorithm(dO_reward_map, dO_size_map):
    car_object_index_map = {}
    index = 0
    all_reward = []
    all_size = []
    for carId in dO_reward_map:
        for objectId, object_reward in enumerate(dO_reward_map[carId]):
            car_object_index_map[index] = [int(carId), objectId]
            all_reward.append(object_reward+0.0001*int(carId)+0.00001*int(objectId))
            all_size.append(dO_size_map[carId][objectId])
            index += 1
    all_reward = np.array(all_reward)
    all_size = np.array(all_size)
    greedy_metric = all_reward
    greedy_index = np.argsort(greedy_metric)[::-1]
    policy_vector = []
    reward_vector = []
    for id in greedy_index:
        policy_vector.append(car_object_index_map[id])
    # transmission_sec
    # print(policy_vector)
    return [policy_vector, reward_vector]

def size_greedy_scheduling_algorithm(dO_reward_map, dO_size_map):
    car_object_index_map = {}
    index = 0
    all_reward = []
    all_size = []
    for carId in dO_reward_map:
        for objectId, object_reward in enumerate(dO_reward_map[carId]):
            car_object_index_map[index] = [int(carId), objectId]
            all_reward.append(object_reward+0.0001*int(carId)+0.00001*int(objectId))
            all_size.append(dO_size_map[carId][objectId])
            index += 1
    all_reward = np.array(all_reward)
    all_size = np.array(all_size)
    greedy_metric = all_size
    greedy_index = np.argsort(greedy_metric)
    policy_vector = []
    reward_vector = []
    for id in greedy_index:
        policy_vector.append(car_object_index_map[id])
    # transmission_sec
    # print(policy_vector)
    return [policy_vector, reward_vector]

def total_reward(dV_vector, received_vector, V, R):
    results_vector = np.multiply(dV_vector,received_vector)
    remaining_reward = np.sum(results_vector)
    # print("Total: ", remaining_reward)
    # f = open("data/total.txt","a")
    # f.write(str(remaining_reward))
    # f.write("\t")
    return remaining_reward

def greedy_scheduling_algorithm(PeerIds, dV_vector, received_vector, V, R, T):
    policy_vector = []
    reward_vector = []
    reward = 0
    previous_reward = 0
    remaining_reward = 0
    for t in range(0, T):
        results_vector = np.multiply(dV_vector,received_vector)
        prioritized_vector = np.sum(results_vector,axis = 0)
        remaining_reward = np.sum(results_vector)
        if remaining_reward == 0:
            break
        if t != 0:
            reward += (previous_reward - remaining_reward)
            reward_vector.append(previous_reward-remaining_reward)
        previous_reward = remaining_reward
        car_view_max = np.unravel_index(np.argmax(prioritized_vector, axis=None), prioritized_vector.shape)
        # print(t, car_view_max)
        policy_vector.append(car_view_max)
        received_vector = transmission(PeerIds, received_vector, car_view_max, V, R)

    results_vector = np.multiply(dV_vector, received_vector)
    # prioritized_vector = np.sum(results_vector, axis = 0)
    remaining_reward = np.sum(results_vector)
    reward += (previous_reward-remaining_reward)
    reward_vector.append(previous_reward - remaining_reward)
    return [policy_vector, reward_vector]

def random_scheduling_algorithm(PeerIds, dV_vector, received_vector, V, R, T):
    policy_vector = []
    reward = 0
    previous_reward = 0
    remaining_reward = 0
    for t in range(0, T):
        results_vector = np.multiply(dV_vector,received_vector)
        prioritized_vector = np.sum(results_vector,axis=0)
        remaining_reward = np.sum(results_vector)
        if t != 0:
            reward += (previous_reward - remaining_reward)
        previous_reward = remaining_reward
        index = np.where(prioritized_vector != 0)
        car_view_max = (0,0)
        if index[0].size != 0 and index[1].size != 0:
            random_index = random.choice(range(0,index[0].size))
            car_view_max = (index[0][random_index],index[1][random_index])
        # print(t, car_view_max)
        policy_vector.append(car_view_max)
        received_vector = transmission(PeerIds, received_vector, car_view_max, V, R)
    results_vector = np.multiply(dV_vector,received_vector)
    # prioritized_vector = np.sum(results_vector, axis=0)
    remaining_reward = np.sum(results_vector)
    reward += (previous_reward - remaining_reward)
    return policy_vector

def roundrobin_scheduling_algorithm(PeerIds, dV_vector, received_vector, V, R, T):
    policy_vector = []
    reward_vector = []
    reward = 0
    previous_reward = 0
    remaining_reward = 0
    PeerIds_Int = [int(s) for s in PeerIds]
    PeerIds_Int = sorted(PeerIds_Int)
    n = 0
    v = 0
    for t in range(0, T):
        results_vector = np.multiply(dV_vector, received_vector)
        prioritized_vector = np.sum(results_vector, axis=0)
        remaining_reward = np.sum(results_vector)
        if remaining_reward == 0:
            break
        if t != 0:
            reward += (previous_reward - remaining_reward)
            reward_vector.append(previous_reward - remaining_reward)
        previous_reward = remaining_reward
        index = np.where(prioritized_vector != 0)
        car_view_max = (PeerIds_Int[n], v)
        # print(t, car_view_max)
        policy_vector.append(car_view_max)
        received_vector = transmission(PeerIds, received_vector, car_view_max, V, R)
        v += 1
        if v >= V:
            v = 0
            n += 1
    results_vector = np.multiply(dV_vector, received_vector)
    # prioritized_vector = np.sum(results_vector, axis=0)
    remaining_reward = np.sum(results_vector)
    reward += (previous_reward - remaining_reward)
    reward_vector.append(previous_reward - remaining_reward)
    return [policy_vector, reward_vector]

def get_success_prob(myTrans, peerTrans):
    d_min = 0.1
    d_max = 100
    p_min = 1
    p_max = 0.75
    peer_x = peerTrans.location.x
    peer_y = peerTrans.location.y
    my_x = myTrans.location.x
    my_y = myTrans.location.y
    distance = [peer_x - my_x, peer_y - my_y]
    d = math.sqrt((distance[0]**2+distance[1]**2))
    ratio = (d-d_min)/(d_max-d_min)
    p = p_min+ratio*(p_max-p_min)
    return p

# assume reachablility test passed somewhere else
def transmission(peers, received_vector, car_view_max, V, R):

    for d_id in peers:
        d_index = int(d_id)
        # if random.random() <= get_success_prob(allplayers[car_view_max[0]].vehicle.get_transform(),d.vehicle.get_transform()):

        # assume probability 1 for now
        received_vector[d_index][car_view_max[0]][car_view_max[1]] = 0
    return received_vector

def checkView(myTrans, peerTrans, length, width, V, V_set):
    """
    This function checks which view is blocked by drawing lines from ego myTrans to 4 courners of peerTrans bounding box
    """
    single_angle = 360 / V
    new_yaw = 0  # peerTrans.rotation.yaw
    peer_range = max(width, length)
    peer_x = peerTrans.location.x
    peer_y = peerTrans.location.y
    my_x = myTrans.location.x
    my_y = myTrans.location.y
    corners = [[peer_x + peer_range / 2 - my_x, peer_y + peer_range / 2 - my_y],
               [peer_x + peer_range / 2 - my_x, peer_y - peer_range / 2 - my_y],
               [peer_x - peer_range / 2 - my_x, peer_y + peer_range / 2 - my_y],
               [peer_x - peer_range / 2 - my_x, peer_y - peer_range / 2 - my_y]]
    view_vector = {}
    for c in corners:
        slope = math.sqrt(c[0] ** 2 + c[1] ** 2)
        corner_angle = np.arccos(c[1] / (slope + 0.00001)) * 180 / np.pi
        if (c[0] >= 0):
            corner_angle = 360 - corner_angle
        for v in V_set:  # range(1,V+1):
            if single_angle * (v - 1) + new_yaw <= corner_angle <= single_angle * v + new_yaw:
                view_vector[v] = 1
                # v1 = v + 1
                # if v1 > V:
                #     v1 = v + 1 - V
                # view_vector[v1] = 1
                # v2 = v - 1
                # if v2 == 0:
                #     v2 = V
                # view_vector[v2] = 1
    return view_vector

def blockage_information(myTrans, peerTrans, length, width, controlPeer, V):
    ### this function is used to determine which views are blocked by vehicles.
    ### [input] myTrans: self information
    ### 	peerTrans: information of the other vehicle
    ###	    	d: information of the other vehicle
    ###	    	V: the total number of views
    ### [output] the index of the blocked views
    V_set = []
    if abs(myTrans.rotation.yaw) < 89.5:
        V_set += [5, 6, 7, 8]
    if abs(myTrans.rotation.yaw) > 90.5:
        V_set += [1, 2, 3, 4]
    if 179.5 > myTrans.rotation.yaw > 0.5:
        V_set += [1, 2, 7, 8]
    if -0.5 > myTrans.rotation.yaw > -179.5:
        V_set += [3, 4, 5, 6]
    V_set = list(set(V_set))
    check_vector = checkView(myTrans, peerTrans, length, width, V, V_set)
    # -1 +1
    view_vector = {}
    for v in check_vector:
        view_vector[v] = 1
        v1 = v + 1
        if v1 > V:
            v1 = v + 1 - V
        view_vector[v1] = 1
        v2 = v - 1
        if v2 == 0:
            v2 = V
        view_vector[v2] = 1
    safety_vector = {}
    for peer in controlPeer:
        twoHopTrans = carla.Transform(carla.Location(x=peer.x,y=peer.y))
        return_vector = checkView(peerTrans, twoHopTrans, peer.DimX, peer.DimY, V, range(1,V+1))
        safety_vector.update(return_vector)
    safety_index = 1
    for key in safety_vector.keys():
        if key in view_vector:
            view_vector[key] += safety_vector[key]*safety_index
        else:
            view_vector[key] = safety_vector[key]*safety_index
    return view_vector


def get_corner_angles(my_x, my_y, corners):

    corner_angles = []
    for corner in corners:
        receiver_to_object_corner_vector = [corner[0] - my_x, corner[1] - my_y]
        euclidean_dist = math.hypot(receiver_to_object_corner_vector[0], receiver_to_object_corner_vector[1])
        corner_angle = np.arccos(receiver_to_object_corner_vector[1] / (euclidean_dist + 0.00001)) * 180 / np.pi
        if (receiver_to_object_corner_vector[0] >= 0):
            corner_angle = 360 - corner_angle
        corner_angles.append(corner_angle)
    return corner_angles

def check_object(myTrans, peerTrans, length, width, ego_detected_object_list, peer_detected_object_list, angle_margin=5, dist_margin=3):
    """
    Ego is the receiver, peer is the sender.
    calculate the angle of object and occlusion angle from receiver (ego), but in the sender's frame of reference.
    TODO: change to ego frame of ref later, for more readabilty

    @param myTrans:
    @param peerTrans:
    @param length:
    @param width:
    @param ego_detected_object_list:
    @param peer_detected_object_list:
    @return:
    """
    ### object_list's coordinate is based on peerTrans....
    object_interest_vector = np.zeros(shape=[len(peer_detected_object_list)])
    peer_range = max(width, length)
    peer_x = 0 #peerTrans.location.x
    peer_y = 0 # peerTrans.location.y
    # my_x = myTrans.location.x
    # my_y = myTrans.location.y
    peer_m= np.matrix([peerTrans.location.x, peerTrans.location.y, peerTrans.location.z])
    peer_m = Utils.world_to_car_transform(peer_m, peerTrans)[0]
    my_m = np.matrix([myTrans.location.x, myTrans.location.y, myTrans.location.z])
    my_m = Utils.world_to_car_transform(my_m, peerTrans)[0]
    my_x = my_m[0]
    my_y = my_m[1]
    # check sender corners
    sender_corners = [[peer_x + peer_range / 2, peer_y + peer_range / 2, 0],
                     [peer_x + peer_range / 2, peer_y - peer_range / 2, 0],
                     [peer_x - peer_range / 2, peer_y + peer_range / 2, 0],
                     [peer_x - peer_range / 2, peer_y - peer_range / 2, 0]]

    # print(sender_corners)
    sender_corner_angles = get_corner_angles(my_x, my_y, sender_corners)
    obstruction_objects_4corners_list = [sender_corners]
    obstruction_objects_4corner_angles_list = [sender_corner_angles]
    obstruction_objects_center_list = [[peer_x,peer_y,0]]
    # print(obstruction_objects_4corner_angles_list)
    # check other ego_detected_obj corners
    for ego_obj in ego_detected_object_list:
        bbox = ego_obj.bounding_box # [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] # center, dimension
        box_x = bbox[0][0]
        box_y = bbox[0][1]
        box_z = bbox[0][2]
        box_ext_x = bbox[1][0] / 2.0
        box_ext_y = bbox[1][1] / 2.0
        box_ext_z = bbox[1][2] / 2.0

        bbox_corners = np.array([[box_x + box_ext_x, box_y + box_ext_y, 0],
                                [box_x + box_ext_x, box_y - box_ext_y, 0],
                                [box_x - box_ext_x, box_y + box_ext_y, 0],
                                [box_x - box_ext_x, box_y - box_ext_y, 0]])

        bbox_corners_peer_perspective = Utils.transform_pointcloud(bbox_corners, peerTrans=myTrans, myTrans=peerTrans)
        bbox_corner_angles = get_corner_angles(my_x, my_y, bbox_corners_peer_perspective.tolist())

        obstruction_objects_4corners_list.append(bbox_corners_peer_perspective.tolist())
        obstruction_objects_4corner_angles_list.append(bbox_corner_angles)
        obstruction_objects_center_list.append(bbox[0])

    # print(obstruction_objects_4corner_angles_list)
    for object_id, deteceted_object in enumerate(peer_detected_object_list):  # range(1,V+1):

        object_center_x = deteceted_object.bounding_box[0][0]  # (x)
        object_center_y = deteceted_object.bounding_box[0][1]  # (y)
        receiver_to_object_center_vector = [peer_x + object_center_x - my_x, peer_y + object_center_y - my_y]
        receiver_to_object_center_distance = math.sqrt(receiver_to_object_center_vector[0] ** 2 + receiver_to_object_center_vector[1] ** 2)
        receiver_to_object_angle = np.arccos(receiver_to_object_center_vector[1] / (receiver_to_object_center_distance + 0.00001)) * 180 / np.pi
        if (receiver_to_object_center_vector[0] >= 0):
            receiver_to_object_angle = 360 - receiver_to_object_angle
        # print("[SCHED DEBUG] obj center {}, dist {}, angle {}".format([object_center_x, object_center_y], receiver_to_object_center_distance, receiver_to_object_angle))

        # check every obstruction, starting from sender itself
        for idx in range(len(obstruction_objects_4corner_angles_list)):

            # check angle
            corner_angles = obstruction_objects_4corner_angles_list[idx]
            object_in_corner_angles = False
            if min(corner_angles) - angle_margin <= receiver_to_object_angle <= max(corner_angles) + angle_margin:
                object_in_corner_angles = True
            # check distance compared to corners
            object_behind_obstruction = True
            corner_dist = []
            for c in obstruction_objects_4corners_list[idx]:
                obstruction_corner_dist = math.sqrt(c[0] ** 2 + c[1] ** 2)
                corner_dist.append(obstruction_corner_dist)
                if receiver_to_object_center_distance < obstruction_corner_dist + dist_margin:
                    object_behind_obstruction = False
                    break
            # print("ego_obj {}: corner range {}, In?({}), distance range {}, behind?({})".format(idx,[min(corner_angles), max(corner_angles)], object_in_corner_angles, [min(corner_dist), max(corner_dist)], object_behind_obstruction))

            # check if same object, break
            obstruction_center_dist = math.hypot(obstruction_objects_center_list[idx][0],
                                                 obstruction_objects_center_list[idx][1])
            # print("center distance: obj {}, ego_obj {}".format(receiver_to_object_center_distance, obstruction_center_dist))
            if abs(receiver_to_object_center_distance - obstruction_center_dist) < dist_margin:
                if object_in_corner_angles:
                    object_interest_vector[object_id] = 0
                    # print("Same obj")
                    break

            if object_behind_obstruction and object_in_corner_angles:
                # print("Diff obj, occluding")
                object_interest_vector[object_id] = 1 # potential interest, may be revoked by other detected object if same

            # print("[SCHED DEBUG] " + str(object_id) + "\t" + str(object_center_x) + "\t" + str(object_center_y) + "\t" + str(object_interest_vector[object_id]))
            # print(object_id, object_center_x, object_center_y, object_interest_vector[object_id])
    return object_interest_vector



#
# def check_object(myTrans, peerTrans, length, width, object_list):
#     ### object_list's coordinate is based on peerTrans....
#     object_interest_vector = np.zeros(shape=[len(object_list)])
#     peer_range = max(width, length)
#     peer_x = 0 #peerTrans.location.x
#     peer_y = 0 # peerTrans.location.y
#     # my_x = myTrans.location.x
#     # my_y = myTrans.location.y
#     peer_m= np.matrix([peerTrans.location.x, peerTrans.location.y, peerTrans.location.z])
#     peer_m = Utils.world_to_car_transform(peer_m, peerTrans)[0]
#     my_m = np.matrix([myTrans.location.x, myTrans.location.y, myTrans.location.z])
#     my_m = Utils.world_to_car_transform(my_m, peerTrans)[0]
#     my_x = my_m[0]
#     my_y = my_m[1]
#     corners = [[peer_x + peer_range / 2 - my_x, peer_y + peer_range / 2 - my_y],
#                [peer_x + peer_range / 2 - my_x, peer_y - peer_range / 2 - my_y],
#                [peer_x - peer_range / 2 - my_x, peer_y + peer_range / 2 - my_y],
#                [peer_x - peer_range / 2 - my_x, peer_y - peer_range / 2 - my_y]]
#     # print(corners)
#     corner_angles = []
#     for c in corners:
#         slope = math.sqrt(c[0] ** 2 + c[1] ** 2)
#         corner_angle = np.arccos(c[1] / (slope + 0.00001)) * 180 / np.pi
#         if (c[0] >= 0):
#             corner_angle = 360 - corner_angle
#         corner_angles.append(corner_angle)
#     for object_id, deteceted_object in enumerate(object_list):  # range(1,V+1):
#         object_center_x = deteceted_object.bounding_box[0][0]  # (x)
#         object_center_y = deteceted_object.bounding_box[0][1]  # (y)
#         object_center = [peer_x + object_center_x - my_x, peer_y + object_center_y - my_y]
#         # print("[SCHED DEBUG] " + str(object_center[0]) + "\t" + str(object_center[1]))
#         # object_center = [object_center_x, object_center_y]
#         object_slope = math.sqrt(object_center[0] ** 2 + object_center[1] ** 2)
#         valid_object_flag = 1
#         for c in corners:
#             slope = math.sqrt(c[0] ** 2 + c[1] ** 2)
#             if object_slope < slope:
#                 valid_object_flag = 0
#                 # print("[ERROR] AN INVALID OBJECT DETECTED")
#         object_angle = np.arccos(object_center[1] / (object_slope + 0.00001)) * 180 / np.pi
#         if (object_center[0] >= 0):
#             object_angle = 360 - object_angle
#         # print(object_angle, min(corner_angles), max(corner_angles))
#         if min(corner_angles) - 5 <= object_angle <= max(corner_angles) + 5 and valid_object_flag:
#             object_interest_vector[object_id] = 1
#         # print("[SCHED DEBUG] " + str(object_id) + "\t" + str(object_center_x) + "\t" + str(object_center_y) + "\t" + str(object_interest_vector[object_id]))
#         # print(object_id, object_center_x, object_center_y, object_interest_vector[object_id])
#     return object_interest_vector

def blockage_information_rect(myTrans, peerTrans, length, width, ego_detected_object_list, peer_detected_object_list):
    V_set = []
    if abs(myTrans.rotation.yaw) < 89.5:
        V_set += [5, 6, 7, 8]
    if abs(myTrans.rotation.yaw) > 90.5:
        V_set += [1, 2, 3, 4]
    if 179.5 > myTrans.rotation.yaw > 0.5:
        V_set += [1, 2, 7, 8]
    if -0.5 > myTrans.rotation.yaw > -179.5:
        V_set += [3, 4, 5, 6]
    V_set = list(set(V_set))
    peer_x = peerTrans.location.x
    peer_y = peerTrans.location.y
    my_x = myTrans.location.x
    my_y = myTrans.location.y
    c = [peer_x - my_x, peer_y - my_y]
    slope = math.sqrt(c[0] ** 2 + c[1] ** 2)
    center_angle = np.arccos(c[1] / (slope + 0.00001)) * 180 / np.pi
    if (c[0] >= 0):
        center_angle = 360 - center_angle
    for v in V_set:  # range(1,V+1):
        if 45 * (v - 1) <= center_angle <= 45 * v:
            object_interest_vector = check_object(myTrans, peerTrans, length, width, ego_detected_object_list, peer_detected_object_list)
            return object_interest_vector

    return np.zeros(shape=[len(peer_detected_object_list)])
