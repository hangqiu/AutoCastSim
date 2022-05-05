import json
from AVR import Utils
import os

class CommLogger(object):
    def __init__(self, _id):
        self.id = _id
        self.trace_id = Utils.EvalEnv.get_trace_id()
        if not Utils.COMMLOG:
            return

        outputdir = "{}/{}/CommLog/".format(Utils.RecordingOutput, str(self.trace_id))
        if not os.path.exists(outputdir):
            print("Creating Logger Directory")
            os.mkdir(outputdir)
        outputdir = outputdir + str(self.id) + '/'
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)

        self.SchedLog = outputdir + _id + "_0_Sched.txt"
        # self.ViewRequestLog = outputdir + "V" + _id + "_ViewRequest.txt"
        self.TXBeaconLog = outputdir + _id + "_1_TXBeacon.txt"
        # self.TXDataLog = outputdir + "V" + _id + "_TXData.txt"
        self.RXBeaconLog = outputdir + _id + "_1_RXBeacon.txt"
        self.RXDataLog = outputdir + _id + "_2_RXData.txt"
        self.SessionStateLog = outputdir + _id + "_1_SessionState.txt"
        self.CoverageLog = outputdir + _id + "_3_Coverage.txt"
        # self.TestLog = outputdir + "V" + _id + "_Test.txt"

        self.SchedWriter = open(self.SchedLog, "w+")
        self.SchedWriter.close()
        # self.ViewRequestWriter = open(self.ViewRequestLog, "w+")
        # self.ViewRequestWriter.close()
        self.TXBeaconWriter = open(self.TXBeaconLog, "w+")
        self.TXBeaconWriter.close()
        # self.TXDataWriter = open(self.TXDataLog, "w+")
        # self.TXDataWriter.close()
        self.RXBeaconWriter = open(self.RXBeaconLog, "w+")
        self.RXBeaconWriter.close()
        self.RXDataWriter = open(self.RXDataLog, "w+")
        self.RXDataWriter.close()
        self.SessionStateWriter = open(self.SessionStateLog, "w+")
        self.SessionStateWriter.close()
        self.CoverageWriter = open(self.CoverageLog, "w+")
        self.CoverageWriter.close()
        # self.TestLogWriter = open(self.TestLog, "w+")
        # self.TestLogWriter.close()

    # def logTest(self, s, frame_count):
    #     if not Utils.COMMLOG:
    #         return
    #     self.TestLogWriter = open(self.TestLog, "a")
    #     self.TestLogWriter.write(str(frame_count) + "\t" + str(s) + "\n")
    #     self.TestLogWriter.close()
    def log_schedule(self, frame_count, schedule, obj_sizes, reward, total_reward, obj_requests, PeerIdMap, dO_reward_map, dO_actorId_map):
        if not Utils.COMMLOG:
            return

        tx_times_ms = []
        tx_times_ms_total = [0]
        obj_size_total = [0]
        for obj_size in obj_sizes:
            time_ms = Utils.transmission_time_sec(obj_size, Utils.Rate) * 1000.0
            tx_times_ms.append(time_ms)
            tx_times_ms_total.append(tx_times_ms_total[-1]+time_ms)
            obj_size_total.append(obj_size_total[-1]+obj_size)
        tx_times_ms_total.pop(0)
        obj_size_total.pop(0)

        msg = {}
        if obj_requests != [] and str(self.id) in PeerIdMap:
            idx = int(PeerIdMap[str(self.id)])
            for d_index, dv in enumerate(obj_requests[idx]):
                for v_index, v in enumerate(dv):
                    if v == 0:
                        continue
                    if d_index not in msg:
                        msg[d_index] = [[v_index, v]]
                    else:
                        msg[d_index].append([v_index, v])
        # print(msg)
        msg = json.dumps(msg, default=lambda o: o.__dict__)

        self.SchedWriter=open(self.SchedLog, "a")
        self.SchedWriter.write(
            "{}\tTotalReward: {}\n\tSchedule(actorid, objid): {}\n\tTime_ms: {}\n\tTime_ms_total: {}\n\tReward: {}\n\tSize: {}\n\tTotal_Size: {}\n".format(
                frame_count, total_reward, schedule, tx_times_ms, tx_times_ms_total, reward, obj_sizes, obj_size_total))

        self.SchedWriter.close()

    def logTXBeacon(self, beacon, frame_count):
        if not Utils.COMMLOG:
            return
        # self.TXBeaconWriter.write(str(frame_count)+ "\n")
        msg = json.dumps(beacon, default=lambda o: o.__dict__)
        self.TXBeaconWriter=open(self.TXBeaconLog, "a")
        self.TXBeaconWriter.write(str(frame_count) + "\t" + msg + "\n")
        # self.TXBeaconWriter.write(str(frame_count) + "\t" + str(beacon) + "\n")
        self.TXBeaconWriter.close()

    # def logTXData(self, data, viewpointsize, objectponitsize, objectIDList):
    #     if not Utils.COMMLOG:
    #         return
    #     # self.TXDataWriter.write(str(frame_count))
    #     msg={}
    #     msg["id"] = data.id
    #     msg["FrameId"] = data.FrameId
    #     msg["ViewID"] = data.ViewID
    #     msg["x"] = data.x
    #     msg["y"] = data.y
    #     msg["z"] = data.z
    #     msg["time_ms"] = data.time_ms
    #     msg["viewpointsize"] = viewpointsize
    #     msg["objectponitsize"] = objectponitsize
    #     msg["objectIDList"] = objectIDList
    #     msg = json.dumps(msg, default=lambda o: o.__dict__)
    #
    #     self.TXDataWriter=open(self.TXDataLog,"a")
    #     self.TXDataWriter.write(msg+ "\n")
    #     self.TXDataWriter.close()

    # def logTXData_FrameCount(self, frame_count):
    #     if not Utils.COMMLOG:
    #         return
    #     pass
    #     # self.TXDataWriter.write(str(frame_count)+ "\n")

    def logRXBeacon(self, beacon, frame_count):
        if not Utils.COMMLOG:
            return
        # self.RXBeaconWriter.write(str(frame_count))
        # msg = json.dumps(beacon, default=lambda o: o.__dict__)
        msg = {}
        msg["id"] = beacon.id
        msg["FrameId"] = beacon.FrameId
        msg["x"] = beacon.x
        msg["y"] = beacon.y
        msg["z"] = beacon.z
        msg["yaw"] = beacon.yaw
        msg["roll"] = beacon.roll
        msg["pitch"] = beacon.pitch
        msg["DimX"] = beacon.DimX
        msg["DimY"] = beacon.DimY
        msg["DimZ"] = beacon.DimZ
        msg["speed"] = beacon.speed
        msg["trigger"] = beacon.trigger

        obj_size = []
        for o in beacon.detected_object_list:
            obj_size.append([o.actor_id, len(o.point_cloud_list)])
        msg["ObjectList"] = obj_size
        msg = json.dumps(msg, default=lambda o: o.__dict__)

        self.RXBeaconWriter=open(self.RXBeaconLog, "a")
        self.RXBeaconWriter.write(str(frame_count) + "\t" + msg + "\n")
        self.RXBeaconWriter.close()

    # def logRXBeacon_FrameCount(self,frame_count):
    #     if not Utils.COMMLOG:
    #         return
    #     pass
    #     # self.RXBeaconWriter.write(str(frame_count)+ "\n")

    def logRXData(self, data):
        if not Utils.COMMLOG:
            return
        msg = {}
        msg["id"] = data.id
        msg["FrameId"] = data.FrameId
        msg["ViewID"] = data.ViewID
        msg["x"] = data.x
        msg["y"] = data.y
        msg["z"] = data.z
        msg["datasize"] = len(data.data)
        msg["time_ms"] = data.time_ms
        # msg["objectIDList"] = data.objectIDList

        msg = json.dumps(msg, default=lambda o: o.__dict__)

        self.RXDataWriter=open(self.RXDataLog, "a")
        self.RXDataWriter.write(msg + "\n")
        self.RXDataWriter.close()

    # def logRXData_FrameCount(self, frame_count):
    #     if not Utils.COMMLOG:
    #         return
    #     pass
    #     # self.RXDataWriter.write(str(frame_count)+ "\n")

    def logSessionState(self, state, frame_count):
        if not Utils.COMMLOG:
            return
        # self.SessionStateWriter.write(str(frame_count)+ "\n")
        msg = {}
        msg["FrameId"] = state.FrameId
        msg["InSession"] = state.InSession
        msg["clusterHeadId"] = state.clusterHeadId
        msg["PeerIds"] = state.PeerIds
        msg = json.dumps(msg, default=lambda o: o.__dict__)

        self.SessionStateWriter = open(self.SessionStateLog, "a")
        self.SessionStateWriter.write(msg+ "\n")
        self.SessionStateWriter.close()

    def logCoverage(self, coverage_count, frame_count):
        if not Utils.COMMLOG:
            return
        self.CoverageWriter = open(self.CoverageLog, "a")
        self.CoverageWriter.write(str(frame_count) + "\t" + str(coverage_count) + "\n")
        self.CoverageWriter.close()
