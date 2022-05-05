# ==============================================================================
# -- Communication Module  -------------------------------------------------------------
# ==============================================================================

import os
import paho.mqtt.client as mqtt
import json
from collections import namedtuple
import threading
from AVR import Utils, Sched
import numpy as np
import carla
import queue


beacon_topic = "Beacon" # must be the same as the data structure name
data_topic = "Data"
broker = "localhost"




#python json stringingfy can't tolerate embbeded objects..... everything to basic type
class Beacon(object):
    def __init__(self, _id, _FrameId, _x, _y, _z, _yaw, _pitch, _roll, _DimX, _DimY, _DimZ, _speed,
                 _object_list, _sensor_view_dict, _sensor_object_dict, _trigger):
        # can't have any functions, must be a clean data structure class for json....
        self.id = _id
        self.topic = beacon_topic
        self.FrameId = _FrameId
        self.x = _x
        self.y = _y
        self.z = _z
        self.yaw = _yaw
        self.roll = _roll
        self.pitch = _pitch
        self.DimX = _DimX
        self.DimY = _DimY
        self.DimZ = _DimZ
        self.speed = [_speed.x, _speed.y, _speed.z]
        self.detected_object_list = _object_list
        # self.cached_sensor_view = _cached_sensor_view
        # self.cached_sensor_object = _cached_sensor_object
        self.cached_sensor_view_size = []
        self.cached_sensor_object_size = []
        for key in _sensor_view_dict:
            self.cached_sensor_view_size.append(_sensor_view_dict[key].shape[0])
        for key in _sensor_object_dict:
            self.cached_sensor_object_size.append(_sensor_object_dict[key].shape[0])
        self.trigger = _trigger


class Data(object):
    def __init__(self, _id, _FrameId, _data, _x, _y, _z, _yaw, _pitch, _roll, _ViewID, _time_ms):
        self.id = _id
        self.topic = data_topic
        self.FrameId = _FrameId
        self.ViewID = _ViewID
        self.data = _data
        self.x = _x
        self.y = _y
        self.z = _z
        self.yaw = _yaw
        self.roll = _roll
        self.pitch = _pitch
        self.time_ms = _time_ms
        # self.objectIDList = _objectIDList
    # can't have any functions, must be a clean data structure class for json....


# Buffer with Lock, Content could be Beacon or Data
class LockBuffer(object):
    def __init__(self, _content):
        self.lock = threading.Lock()
        self.content = _content

    def get_content(self):
        self.lock.acquire()
        c = self.content
        self.lock.release()
        return c

    def set_content(self,_content):
        self.lock.acquire()

        if self.content.topic == beacon_topic:
            self.content = _content

        if self.content.topic == data_topic:
            if self.content.FrameId < _content.FrameId:
                self.content = _content
            else:
                if self.content.FrameId == _content.FrameId:
                    # merge new views
                    self.content.data.extend(_content.data)

        self.lock.release()


class SessionState(object):
    def __init__(self):
        self.InSessionLock = threading.Lock()
        self.FrameId = -1
        self.InSession = False
        self.clusterHeadId = None
        self.PeerIds = []

    def IsInSession(self):
        self.InSessionLock.acquire()
        InSession = self.InSession
        self.InSessionLock.release()
        return InSession

    def get_SessionState(self):
        self.InSessionLock.acquire()
        sess = SessionState()
        sess.InScheduleSession = self.InSession
        sess.clusterHeadId = self.clusterHeadId
        sess.FrameId = self.FrameId
        sess.PeerIds = self.PeerIds
        self.InSessionLock.release()
        return sess

    def try_SessionBegin(self,_clusterHead, _FrameID, _PeerIds):
        succ = False
        # print("Try being session")
        self.InSessionLock.acquire()
        if self.InSession == False:
            self.InSession = True
            self.clusterHeadId = _clusterHead
            self.FrameId = _FrameID
            self.PeerIds = _PeerIds
            succ = True
            # print("New Session Success")
        # else:
            # print(str(self.InSession))
        self.InSessionLock.release()
        return succ

    def end_Session(self):
        self.InSessionLock.acquire()
        self.InSession = False
        self.InSessionLock.release()


class Comm(object):
    def __init__(self, _parent_collaborator, _topic):
        self.parent_collaborator = _parent_collaborator
        self.id = self.parent_collaborator.id
        self.topic = _topic
        self.buffer = dict()  # a dictionary of latest rx content, indexed by peer id
        self.queue = queue.Queue()

        self.sess = SessionState()
        self.scheduler = Sched.Sched(self)

        self.client = mqtt.Client(userdata=self)
        self.client.on_connect = on_connect
        self.client.on_message = on_message
        self.client.connect(broker, port=Utils.mqtt_port)
        self.client.loop_start()
        self.client.subscribe(self.topic, qos=2)
        self.destroyed = False

        self.process = threading.Thread(target=self.processMsgs)
        self.process.start()

    def __del__(self):
        self.destroy()

    def destroy(self):
        self.client.loop_stop()
        self.client.disconnect()
        self.destroyed = True
        # print("waiting for comm to join")
        self.process.join()
        print("Destoryed Comm {}-{}".format(self.id, self.topic))

    def pub_msg(self, content):
        msg = json.dumps(content, default=lambda o: o.__dict__)
        # print(str(self.id) + ": Publishing " + self.topic)
        self.client.publish(self.topic, msg, qos=2)

    def pub_ToBeaconList(self, beacon):
        self.parent_collaborator.agent_wrapper.append_beacon_list(beacon)

    def get_carla_Transform_from_Beacon(self, beacon):
        return carla.Transform(carla.Location(x=beacon.x,y=beacon.y,z=beacon.z),carla.Rotation(yaw=beacon.yaw,pitch=beacon.pitch,roll=beacon.roll))

    def get_carla_Transform_from_Data(self, data):
        return carla.Transform(carla.Location(x=data.x, y=data.y, z=data.z), carla.Rotation(yaw=data.yaw, pitch=data.pitch, roll=data.roll))

    def get_ndarray_from_Data(self, d):
        return np.array(d.data)

    def get_carla_Transform_LocationOnly_from_Data(self,d):
        return carla.Transform(carla.Location(x=d.x,y=d.y,z=d.z))

    def saveMsgToPeerBuffer(self, _content):
        # Save message to buffer
        PeerId = str(_content.id)
        if (PeerId in self.buffer.keys()):
            self.buffer[PeerId].set_content(_content)
        else:
            self.buffer[PeerId] = LockBuffer(_content)


    def DistanceCheck(self, _content):
        if self.id != _content.id:
            # print("receive others")
            if str(self.id) in self.buffer.keys():
                # print("self already in")
                # Reachability Check
                mData = self.buffer[str(self.id)].get_content()
                mTrans = carla.Transform(carla.Location(x=mData.x, y=mData.y, z=mData.z))
                peerTrans = carla.Transform(carla.Location(x=_content.x, y=_content.y, z=_content.z))
                if not Utils.reachable_check(mTrans, peerTrans, 2 * Utils.HalfRadioRange):
                    # print("Out of Range")
                    return False
            else:
                # print("self not in yet")
                return False

        return True

    def processMsgs(self):
        # print("Begin Processing Queue")
        Debug = False
        while True:
            if self.destroyed:
                break
            try:
                _content = self.queue.get(timeout=3)
            except queue.Empty:
                continue

            self.queue.task_done()
            if Debug:
                print(str(self.id) + ": Received " + self.topic + " frame " +
                      str(_content.FrameId) + " from " + str(_content.id))
            # print("get one")
            InRange = self.DistanceCheck(_content)
            if not InRange:
                continue

            # Save message to buffer
            self.saveMsgToPeerBuffer(_content)

            if Debug:
                print(str(self.id) +":" + self.topic + " " + str(self.buffer.keys()))

            ### Data should check if buffer filled and current session ends
            if self.topic == data_topic:
                self.parent_collaborator.logger.logRXData(_content)
                if Debug:
                    print("Check for Data")
                sess = self.parent_collaborator.ControlChannel.sess.get_SessionState()
                if sess.InScheduleSession == False:
                    if Debug:
                        print("Not In Session")
                    continue

                endSession = True
                for PeerId in self.buffer.keys():
                    c = self.buffer[PeerId].get_content()
                    if c.FrameId < sess.FrameId:
                        # print(sess.FrameId)
                        # print(c.id)
                        endSession = False
                        break
                if endSession:
                    if Debug:
                        print(str(self.id) + ": All data collected! Frame " + str(sess.FrameId) + " Session Ends")
                    self.parent_collaborator.ControlChannel.sess.end_Session()
                else:
                    if Debug:
                        print("Missing Data in this Session")


# The callback for when the client receives a CONNACK response from the server.
def on_connect(c, mComm, flags, rc):
    print(str(mComm.id) + " Connected with result code " + str(rc))
    print(str(mComm.id) + " Subscribe to " + mComm.topic)
    # c.subscribe(userdata.topic)


# The callback for when a PUBLISH message is received from the server.
def on_message(c, mComm, msg):
    m = msg.payload.decode("utf-8")
    _content = json.loads(m, object_hook=lambda d: namedtuple(mComm.topic, d.keys())(*d.values()))
    # print(str(mComm.id) + ": Received " + mComm.topic +" frame " + str(_content.FrameId) + " from " + str(_content.id))
    mComm.queue.put(_content, block=False)
