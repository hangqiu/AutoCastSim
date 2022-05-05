import copy
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from AVR import Utils


class LogParser(object):
    def __init__(self, trace_path_list, meta_filepath_list, name_list):
        self.name_list = name_list
        self.trace_path_list = trace_path_list
        self.commlog_list = []
        for trace_path in trace_path_list:
            self.commlog_list.append(trace_path + "/CommLog/")
        self.metadata_path_list = meta_filepath_list

    def analyze_frame_visibility(self, jsonfile):
        data = json.load(jsonfile)
        ego_trans = data['ego_vehicle_transform']
        ego_x = ego_trans['x']
        ego_y = ego_trans['y']
        actors = data['other_actors']
        ratio_dict = dict()
        single_points_dict = dict()
        shared_points_dict = dict()

        other_single_points_dict = dict()
        other_shared_points_dict = dict()

        for actor_id in actors:
            actor_data = actors[actor_id]
            if 'vehicle' not in actor_data['type']:
                continue
            single_points = actor_data['detected_quantpoints']
            shared_points = actor_data['detected_quantpoints_after_sharing']

            single_points_dict[actor_id] = single_points
            shared_points_dict[actor_id] = shared_points

            if "other_single_detected_objects" in actor_data:
                other_single_points_dict[actor_id] = actor_data['other_single_detected_objects']
            if "other_shared_detected_objects" in actor_data:
                other_shared_points_dict[actor_id] = actor_data["other_shared_detected_objects"]

            # print(f"Actor: {actor_id}, single_points: {single_points}, shared_points: {shared_points}")

            ratio = -1
            if shared_points != 0:
                ratio = single_points / shared_points

            actor_trans = actor_data['transform']
            actor_x = actor_trans['x']
            actor_y = actor_trans['y']
            dist = Utils.get_distance(x1=ego_x, y1=ego_y, x2=actor_x, y2=actor_y)

            if dist > 50:
                ratio = -1
            ratio_dict[actor_id] = ratio
            # weight by distance
            # ratio_dict[actor_id] = ratio / dist

        return ratio_dict, single_points_dict, shared_points_dict, other_shared_points_dict, other_shared_points_dict

    def analyze_visibility(self, trace_path, metadata_path, objID=None, collision_frame_window=None, plot=False):

        files = sorted(os.listdir(metadata_path))
        detected_obj_ratio_dict = dict()
        detected_obj_single_points_dict = dict()
        detected_obj_shared_points_dict = dict()
        for j in range(len(files)):
            fp = metadata_path + files[j]
            # print(fp)
            f = open(fp, 'r')
            ratio_dict, single_points_dict, shared_points_dict, other_single_detected_objects, other_shared_detected_objects = self.analyze_frame_visibility(
                f)
            for actor_id in ratio_dict:
                if actor_id not in detected_obj_ratio_dict:
                    detected_obj_ratio_dict[actor_id] = dict()
                    detected_obj_single_points_dict[actor_id] = dict()
                    detected_obj_shared_points_dict[actor_id] = dict()

                # add filter for relevance: occlusion and distance (optional)
                if single_points_dict[actor_id] < 10:
                    detected_obj_ratio_dict[actor_id][j] = ratio_dict[actor_id]
                    detected_obj_single_points_dict[actor_id][j] = single_points_dict[actor_id]
                    detected_obj_shared_points_dict[actor_id][j] = shared_points_dict[actor_id]

        detected_obj_ratio_pd = pd.DataFrame(detected_obj_ratio_dict)
        detected_obj_single_points_pd = pd.DataFrame(detected_obj_single_points_dict)
        detected_obj_shared_points_pd = pd.DataFrame(detected_obj_shared_points_dict)
        # print(detected_obj_ratio_pd)
        # print(detected_obj_single_points_pd)
        # print(detected_obj_shared_points_pd)

        avg_ratio_pd = self.plot_per_obj_avg_over_frame(detected_obj_ratio_dict, trace_path + "/visibility.png",
                                                        value_name='Visibility(Single/Shared)')

        if plot:
            self.plot_per_obj(detected_obj_ratio_pd, trace_path + "/per_obj_ratio.png")
            self.plot_per_obj(detected_obj_single_points_pd, trace_path + "/per_obj_single.png")
            self.plot_per_obj(detected_obj_shared_points_pd, trace_path + "/per_obj_shared.png")
            if objID is not None:
                self.plot_per_obj(detected_obj_single_points_pd, trace_path + f"/per_obj_single_{objID}.png", objId=objID)
                self.plot_per_obj(detected_obj_shared_points_pd, trace_path + f"/per_obj_shared_{objID}.png", objId=objID)



            obj_points_pd = self.plot_per_frame_avg_over_obj(detected_obj_points_pd_list=[detected_obj_single_points_pd,
                                                                                          detected_obj_shared_points_pd],
                                                             value_name_list=["Single", "Shared"],
                                                             collision_frame_window=collision_frame_window,
                                                             path=trace_path + "/per_frame_compare.png"
                                                         )
            target_obj_points_pd = self.plot_per_frame_avg_over_obj(
                detected_obj_points_pd_list=[detected_obj_single_points_pd[objID],
                                             detected_obj_shared_points_pd[objID]],
                value_name_list=["Single", "Shared"],
                collision_frame_window=collision_frame_window,
                path=trace_path + f"/per_frame_compare_{objID}.png"
            )

        return avg_ratio_pd, detected_obj_single_points_pd, detected_obj_shared_points_pd

    def plot_per_frame_avg_over_obj(self, detected_obj_points_pd_list, value_name_list, collision_frame_window, path):
        obj_points_pd = pd.DataFrame()
        min_rows = None
        for i in range(len(value_name_list)):
            detected_obj_points_pd = detected_obj_points_pd_list[i]
            num_rows = detected_obj_points_pd.shape[0]
            if min_rows is None:
                min_rows = num_rows
            elif num_rows < min_rows:
                min_rows = num_rows
        for i in range(len(value_name_list)):
            detected_obj_points_pd = detected_obj_points_pd_list[i]
            detected_obj_points_pd = detected_obj_points_pd[:min_rows]
            detected_obj_points_pd = detected_obj_points_pd.reset_index().rename(columns={'index': 'Frame ID'})
            # print(detected_obj_points_pd)
            detected_obj_points_pd = detected_obj_points_pd.melt(id_vars='Frame ID',
                                                                 var_name=f"Object_ID_{value_name_list[i]}",
                                                                 value_name=value_name_list[i])
            # print(detected_obj_points_pd)
            detected_obj_points_pd = detected_obj_points_pd.rename(
                columns={'Frame ID': f"Frame_ID_{value_name_list[i]}"})
            # print(detected_obj_points_pd)
            obj_points_pd = pd.concat([obj_points_pd, detected_obj_points_pd], axis=1)
        # print(obj_points_pd)

        col_list = copy.deepcopy(value_name_list)
        col_list.extend([
            f"Object_ID_{value_name_list[0]}",
            f"Frame_ID_{value_name_list[0]}"
        ])
        # print(obj_points_pd)
        # print(col_list)
        obj_points_pd = obj_points_pd[col_list].rename(columns={f"Frame_ID_{value_name_list[0]}": "Frame ID",
                                                                f"Object_ID_{value_name_list[0]}": "Object ID"})
        # obj_points_pd = obj_points_pd.reset_index()
        # col_list = ["Frame ID"] + col_list[:-1] + ["Object ID"]
        # obj_points_pd.columns = col_list
        # print(obj_points_pd)
        obj_points_pd = obj_points_pd.melt(id_vars=["Frame ID", "Object ID"], var_name="mode",
                                           value_name="Object Points")
        # print(obj_points_pd)

        plt.figure()
        sns.lineplot(data=obj_points_pd, x="Frame ID", y="Object Points", hue="mode")
        plt.xlabel("Frame ID")
        plt.ylabel("# of visible cubic (5x5x5cm)")
        if collision_frame_window is not None:
            plt.xlim(collision_frame_window)
        plt.savefig(path)
        plt.close()

        return obj_points_pd

    def plot_per_obj_avg_over_frame(self, detected_obj_ratio_dict, path, value_name):
        avg_ratio = dict()
        for actor_id in detected_obj_ratio_dict:
            vector = detected_obj_ratio_dict[actor_id]
            avg = np.mean(list(vector.values()))
            avg_ratio[actor_id] = avg

        # print(avg_ratio)
        avg_ratio_pd = pd.DataFrame(avg_ratio, index=[0])
        avg_ratio_pd = avg_ratio_pd.melt()
        avg_ratio_pd.columns = ['ObjectID', value_name]
        # print(avg_ratio_pd)

        plt.figure()
        sns.barplot(x='ObjectID', y=value_name, data=avg_ratio_pd)
        plt.savefig(path)
        plt.close()
        return avg_ratio_pd

    def plot_per_obj(self, detected_obj_pd, path, objId=None):
        # print(detected_obj_pd)
        if objId is not None:
            # print(detected_obj_pd.keys())
            detected_obj_pd = detected_obj_pd[objId]

        plt.figure()
        # sns.barplot(x='FrameID', y='Visibility(Single/Shared)', data=detected_obj_ratio_pd)
        sns.lineplot(data=detected_obj_pd)
        plt.xlabel("Frame ID")
        plt.ylabel("# of visible cubic (5x5x5cm)")
        plt.legend(title="Object ID")
        plt.savefig(path)
        plt.close()

    def compare_trace_detection_ratio(self, compare_name, collision_frame_window=None, target_objid_list=None):
        slice_columns = []
        obj_points_pd_list = []
        target_obj_points_pd_list = []
        avg_ratio_pd_concat = None
        for i in range(len(self.trace_path_list)):
            trace_path = self.trace_path_list[i]
            meta_path = self.metadata_path_list[i]
            if target_objid_list is not None:
                avg_ratio_pd, _, obj_points_pd = self.analyze_visibility(trace_path, meta_path, target_objid_list[i])
                target_obj_points_pd_list.append(obj_points_pd[target_objid_list[i]])
            else:
                avg_ratio_pd, _, obj_points_pd = self.analyze_visibility(trace_path, meta_path)
            avg_ratio_pd.columns = [f"{self.name_list[i]}_ObjID", f"{self.name_list[i]}_Ratio"]
            if i == 0:
                avg_ratio_pd_concat = avg_ratio_pd
            else:
                avg_ratio_pd_concat = pd.concat([avg_ratio_pd_concat, avg_ratio_pd], axis=1)
            obj_points_pd_list.append(obj_points_pd)
            slice_columns.append(f"{self.name_list[i]}_Ratio")

        self.plot_per_frame_avg_over_obj(detected_obj_points_pd_list=obj_points_pd_list,
                                         value_name_list=self.name_list,
                                         collision_frame_window=collision_frame_window,
                                         path=f"./analysis/{compare_name}_points_compare.png")

        # importance filter
        autocast_obj_points = obj_points_pd_list[1]
        # print(autocast_obj_points)
        agnostic_obj_points = obj_points_pd_list[0]
        # print(agnostic_obj_points)
        mask = autocast_obj_points.mask(autocast_obj_points>0, other=1)
        # print(mask)
        ag_cols = agnostic_obj_points.columns
        agnostic_obj_points.columns = autocast_obj_points.columns
        agnostic_obj_points = agnostic_obj_points * mask
        # print(agnostic_obj_points)
        agnostic_obj_points.columns = ag_cols
        # raise
        self.plot_per_frame_avg_over_obj(detected_obj_points_pd_list=[autocast_obj_points,agnostic_obj_points],
                                         value_name_list=['AutoCast', 'Agnostic'],
                                         collision_frame_window=collision_frame_window,
                                         path=f"./analysis/{compare_name}_filtered_points_compare.png")


        if target_objid_list is not None:
            self.plot_per_frame_avg_over_obj(detected_obj_points_pd_list=target_obj_points_pd_list,
                                             value_name_list=self.name_list,
                                             collision_frame_window=collision_frame_window,
                                             path=f"./analysis/{compare_name}_points_compare_target_obj.png")

        avg_ratio_pd_concat = avg_ratio_pd_concat[slice_columns]
        avg_ratio_pd_concat.columns = self.name_list
        # print(avg_ratio_pd_concat)
        self.plot_ratio_compare(avg_ratio_pd_concat, path=f"./analysis/{compare_name}_visibility_compare.png")

    def plot_ratio_compare(self, avg_ratio_pd_concat, path):
        avg_ratio_pd_concat = avg_ratio_pd_concat.reset_index()
        # print(avg_ratio_pd_concat)
        x_name = "Object ID"
        hue_name = "mode"
        y_name = "Occlusion Ratio"
        avg_ratio_pd_concat = avg_ratio_pd_concat.melt(
            id_vars="index",
            var_name="mode",
            value_name="occlusion ratio"
        )
        avg_ratio_pd_concat.columns = [x_name, hue_name, y_name]
        # print(avg_ratio_pd_concat)

        plt.figure()
        sns.lineplot(x=x_name, y=y_name, hue=hue_name, data=avg_ratio_pd_concat)
        plt.savefig(path)
        plt.close()
        return avg_ratio_pd_concat

    def compare_action(self, compare_name, name_list, collision_frame_window):
        throttle_list = []
        brake_list = []
        steer_list = []
        for i in range(len(self.trace_path_list)):
            trace_path = self.trace_path_list[i]
            meta_path = self.metadata_path_list[i]

            files = sorted(os.listdir(meta_path))
            throttle = []
            brake = []
            steer = []
            for j in range(len(files)):
                fp = meta_path + files[j]
                # print(fp)
                f = open(fp, 'r')
                data = json.load(f)
                ego_throttle = data['throttle']
                ego_brake = data['brake']
                ego_steer = data['steer']
                throttle.append(ego_throttle)
                brake.append(ego_brake)
                steer.append(ego_steer)
            throttle_list.append(throttle)
            brake_list.append(brake)
            steer_list.append(steer)
        self.plot_action(throttle_list, 'throttle')
        self.plot_action(brake_list, 'brake')
        self.plot_action(steer_list, 'stee')

    def plot_action(self, throttle_list, action_name):
        throttle_pd = pd.DataFrame(throttle_list)
        throttle_pd = throttle_pd.transpose()
        throttle_pd.columns = name_list
        # print(throttle_pd)
        plt.figure()
        sns.lineplot(data=throttle_pd)
        plt.xlim(collision_frame_window)
        plt.savefig(f"./analysis/{compare_name}_{action_name}_compare.png")
        plt.close()


def test_trace_config():
    target_objid_list = None
    # autocast_trace_id = 17
    # agnostic_trace_id = 18
    # autocast_trace_id = 28
    # agnostic_trace_id = 27
    # autocast_trace_id = "00"
    # agnostic_trace_id = "01"
    autocast_trace_id = 11
    agnostic_trace_id = 10
    compare_name = "Test_AutoCast_vs_Agnostic"

    # autocast_trace_id = 19
    # agnostic_trace_id = 20
    # compare_name = "Scen10_BG60_AutoCast_vs_Agnostic"
    # target_objid_list = [312, 370]

    # autocast_trace_id = 21
    # autocast_trace_id = "02"
    # agnostic_trace_id = 22
    # compare_name = "Scen10_BG110_AutoCast_vs_Agnostic"

    # autocast_trace_id = 24
    # agnostic_trace_id = 25
    # compare_name = "Scen10_BG30_AutoCast_vs_Agnostic"

    trace_path_list = [f"test_traces/scalability/{autocast_trace_id}",
                       f"test_traces/scalability/{agnostic_trace_id}"]
    meta_path_list = [f"test_traces/scalability/{autocast_trace_id}/episode_000{autocast_trace_id}/measurements/",
                      f"test_traces/scalability/{agnostic_trace_id}/episode_000{agnostic_trace_id}/measurements/"]


if __name__ == "__main__":
    # target_objid_list = None
    trace_path_list = ["Scen10_Share_30_Agnostic/0",
                       "Scen10_Share_30_NoExtrap/0"]
    meta_path_list = ["Scen10_Share_30_Agnostic/0/episode_00000/measurements/",
                      "Scen10_Share_30_NoExtrap/0/episode_00000/measurements/"]
    compare_name = "Scen10_BG120_AutoCast_vs_Agnostic"
    target_objid_list = ['469', '173']
    collision_frame_window = [130, 170]

    name_list = ["Agnostic", "AutoCast"]
    # mLogs = LogParser("test_traces/17", "test_traces/17/episode_00017/measurements/")
    # mLogs = LogParser("test_traces/18", "test_traces/18/episode_00018/measurements/")
    mLogs = LogParser(trace_path_list, meta_path_list, name_list)
    # mLogs.analyze_visibility()
    mLogs.compare_action(compare_name, name_list, collision_frame_window)
    mLogs.compare_trace_detection_ratio(compare_name=compare_name, collision_frame_window=collision_frame_window,
                                        target_objid_list=target_objid_list)
