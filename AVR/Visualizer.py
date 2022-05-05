import glob
import json
import os.path
import time

import carla
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from AVR import Utils

# 45 degree tilted angle constants
# get these parameters using the o3d UI, ctrl x after adjusting to the right angle
front = np.array([0.64157564667260192, 0.62421841289783264, -0.44579374445582315])
lookat = np.array([2.5232705666375206, 3.0640390293142192, -1.1871332121214975])
up = np.array([-0.16202590518996754, -0.45777530006724931, -0.87417926119058198])
zoom = 0.08
# the bounding box decides the view, even given the same front lookat up and zoom, if boundingbox is different view is different
# from point cloud. get_max_bound()
# workaround append max min to the pcd and crop it to the max min
boundingbox_maxmin_np = np.array([[100, 100, 8.0], [-100, -100, -4.0]])


# boundingbox_max= np.array([ 49.426689147949219, 49.454471588134766, 7.7580423355102539 ])
# boundingbox_min= np.array([-23.065624237060547, -48.645954132080078, -3.3848309516906738 ])


class Visualizer(object):
    # o3d default visualization angle
    front = np.array([0.0, 0.0, 1.0])
    lookat = np.array([13.198856353759766, -3.9548139624880179, 2.1813147068023682])
    up = np.array([0.0, 1.0, 0.0])
    zoom = 0.7

    @staticmethod
    def np_to_pcd_with_fixed_bbox(pcd_np, boundingbox_maxmin_np, color = None):
        bbox_pcd = o3d.geometry.PointCloud()
        bbox_pcd.points = o3d.utility.Vector3dVector(boundingbox_maxmin_np)
        bbox = bbox_pcd.get_axis_aligned_bounding_box()
        # print(bbox)

        pcd_np = np.concatenate([pcd_np, boundingbox_maxmin_np])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        print(pcd)
        pcd_crop = pcd.crop(bbox)
        print(pcd_crop)

        if color is not None:
            pcd_crop.colors = o3d.utility.Vector3dVector(color)
        # print(pcd_crop.get_axis_aligned_bounding_box())
        # print(pcd_crop.get_min_bound())
        # print(pcd_crop.get_max_bound())
        return pcd_crop

    @staticmethod
    def visualize_pointcloud(lidar_np, front=None,
                             lookat=None,
                             zoom=None,
                             up=None,
                             ):
        """
        Visualize point cloud in 3D from a particular angel of view
        """
        if front is None: _front = Visualizer.front
        if lookat is None: lookat = Visualizer.lookat
        if zoom is None: zoom = Visualizer.zoom
        if up is None: up = Visualizer.up
        # if boundingbox_max is None: boundingbox_max = Visualizer.boundingbox_max
        # if boundingbox_min is None: boundingbox_min = Visualizer.boundingbox_min

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(lidar_np)

        pcd = Visualizer.np_to_pcd_with_fixed_bbox(lidar_np, boundingbox_maxmin_np)

        o3d.visualization.draw_geometries(
            [pcd], front=front, lookat=lookat, zoom=zoom, up=up)

    @staticmethod
    def save_o3d_visualize_pointcloud(lidar_np,
                                      output_path,
                                      front=None,
                                      lookat=None,
                                      zoom=None,
                                      up=None,
                                      ):
        """
        Save point cloud in 3D from a particular angel of view
        reference:http://www.open3d.org/docs/release/tutorial/visualization/customized_visualization.html#capture-images-in-a-customized-animation
        """
        if front is None: _front = Visualizer.front
        if lookat is None: lookat = Visualizer.lookat
        if zoom is None: zoom = Visualizer.zoom
        if up is None: up = Visualizer.up

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(lidar_np)
        pcd = Visualizer.np_to_pcd_with_fixed_bbox(lidar_np, boundingbox_maxmin_np)

        # reproduce the following function in o3d vis:
        #       http://www.open3d.org/docs/release/tutorial/visualization/customized_visualization.html?highlight=rotate_view
        # o3d.visualization.draw_geometries(
        #     [pcd], front=front, lookat=lookat, zoom=zoom, up=up)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        # rotate camera
        ctr = vis.get_view_control()
        ctr.set_front(front)
        ctr.set_lookat(lookat)
        ctr.set_up(up)
        ctr.set_zoom(zoom)
        # change_background_to_black
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        # Save the image
        # vis.run()
        # vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(False)
        plt.imsave(output_path, np.asarray(image), dpi=1)
        vis.destroy_window()

    @staticmethod
    def save_o3d_visualize_fused_pointcloud(lidar_np,
                                            pcd_shared_np,
                                            output_path,
                                            front=None,
                                            lookat=None,
                                            zoom=None,
                                            up=None,
                                            ):
        """
        Visualize point cloud in 3D from a particular angel of view
        reference:
            customized visualizatoin: http://www.open3d.org/docs/release/tutorial/visualization/customized_visualization.html#capture-images-in-a-customized-animation

        """
        if front is None: _front = Visualizer.front
        if lookat is None: lookat = Visualizer.lookat
        if zoom is None: zoom = Visualizer.zoom
        if up is None: up = Visualizer.up

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(lidar_np)
        pcd = Visualizer.np_to_pcd_with_fixed_bbox(lidar_np, boundingbox_maxmin_np)

        # assign special color to shared points
        # pcd_shared = o3d.geometry.PointCloud()
        # pcd_shared.points = o3d.utility.Vector3dVector(pcd_shared_np)


        pcd_shared = Visualizer.np_to_pcd_with_fixed_bbox(pcd_shared_np, boundingbox_maxmin_np)
        size = np.asarray(pcd_shared.points).shape[0]
        colors = np.ones((size, 1)) * 255
        colors = np.concatenate([colors, np.zeros((size, 1)), np.zeros((size, 1))], axis=1)

        # colors = colors.transpose()
        # print(colors)
        # print(colors.shape)
        pcd_shared.colors = o3d.utility.Vector3dVector(colors)

        # print(pcd_shared.colors)

        # reproduce the following function in o3d vis:
        #       http://www.open3d.org/docs/release/tutorial/visualization/customized_visualization.html?highlight=rotate_view
        # Non - blocking view: http: // www.open3d.org / docs / release / tutorial / visualization / non_blocking_visualization.html
        # o3d.visualization.draw_geometries(
        #     [pcd_shared], front=front, lookat=lookat, zoom=zoom, up=up)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.add_geometry(pcd_shared)
        # rotate camera
        ctr = vis.get_view_control()
        ctr.set_front(front)
        ctr.set_lookat(lookat)
        ctr.set_up(up)
        ctr.set_zoom(zoom)
        # change_background_to_black
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        # Save the image
        # vis.run()
        # vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(False)
        plt.imsave(output_path, np.asarray(image), dpi=1)
        vis.destroy_window()

    @staticmethod
    def convert_pointcloud_into_3D_video(
            data_path, output_path,
            index_path=None,
            json_meta_filepath=None,
            fix_view_angle=False,
            front=None, lookat=None, zoom=None, up=None,
    ):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_names = data_path + "/" + "*.npy"
        files = glob.glob(file_names)

        files.sort()
        initial_trans = None
        for file in files:
            pcd_np = np.load(file)
            file_name = file.split('/')[-1].split('.')[0]
            pcd_shared_index = pcd_np.shape[0]
            if index_path is not None:
                pcd_shared_index = np.load(index_path + '/' + file_name + ".npy")
            if fix_view_angle:
                #  convert to not spining angle using the initial car's yaw pitch roll
                jsonpath = json_meta_filepath + "/" + file_name + ".json"
                jsonfile = open(jsonpath)
                # print(jsonpath)
                meta_data = json.load(jsonfile)
                jsonfile.close()
                ego_trans = meta_data["ego_vehicle_transform"]
                if initial_trans is None:
                    initial_trans = ego_trans
                ego_carla_trans = carla.Transform(
                    carla.Location(x=ego_trans["x"], y=ego_trans["y"], z=ego_trans["z"]),
                    carla.Rotation(yaw=ego_trans["yaw"], pitch=ego_trans["pitch"], roll=ego_trans["roll"])
                )
                modified_ego_carla_trans = carla.Transform(
                    carla.Location(x=ego_trans["x"], y=ego_trans["y"], z=ego_trans["z"]),
                    carla.Rotation(yaw=initial_trans["yaw"], pitch=initial_trans["pitch"], roll=initial_trans["roll"])
                )
                pcd_np = Utils.transform_pointcloud(pcd_np, ego_carla_trans, modified_ego_carla_trans)

            if index_path is not None:
                pcd_shared_np = pcd_np[pcd_shared_index:]
                pcd_np = pcd_np[0:pcd_shared_index]
                Visualizer.save_o3d_visualize_fused_pointcloud(
                    pcd_np, pcd_shared_np,
                    output_path=output_path + '/' + file_name + '.png',
                    front=front, lookat=lookat, zoom=zoom, up=up
                )
            else:
                Visualizer.save_o3d_visualize_pointcloud(
                    pcd_np,
                    output_path=output_path + '/' + file_name + '.png',
                    front=front, lookat=lookat, zoom=zoom, up=up
                )

        Utils.compile_video(output_path, output_name="0_3D_PointCloud.mp4")


def test_vis_point_cloud():
    # # test vis point cloud
    # pcd_path = "data/3DView/1/episode_00001/544_LIDAR/11129.npy"
    pcd_path = "data/CoopernautVideoData/6_ Overtaking/CPT/2/259_LIDARFused/00139.npy"
    pcd_np = np.load(pcd_path)
    Visualizer.visualize_pointcloud(
        pcd_np,
        front=front, lookat=lookat, zoom=zoom, up=up,
    )


def test_save_point_cloud():
    # # test vis point cloud
    pcd_path = "data/3DView/1/episode_00001/544_LIDAR/11128.npy"
    pcd_np = np.load(pcd_path)
    Visualizer.save_o3d_visualize_pointcloud(
        pcd_np, output_path="tmp.png",
        front=front, lookat=lookat, zoom=zoom, up=up,
    )


def test_save_shared_point_cloud():
    pcd_path = "data/CoopernautVideoData/6_ Overtaking/CPT/2/259_LIDARFused/00139.npy"
    pcd_np = np.load(pcd_path)
    Visualizer.save_o3d_visualize_pointcloud(
        pcd_np, output_path="tmp_ego.png",
        front=front, lookat=lookat, zoom=zoom, up=up,
    )

    pcd_shared_index_path = "data/CoopernautVideoData/6_ Overtaking/CPT/2/259_LIDARIndex/00139.npy"
    pcd_shared_index = np.load(pcd_shared_index_path)
    # pcd_shared_index = pcd_shared_index[0]
    # print(pcd_np.shape)
    # print(pcd_shared_index)
    pcd_shared_np = pcd_np[pcd_shared_index:]
    pcd_np = pcd_np[0:pcd_shared_index]
    Visualizer.save_o3d_visualize_pointcloud(
        pcd_shared_np, output_path="tmp_shared.png",
        front=front, lookat=lookat, zoom=zoom, up=up,
    )
    Visualizer.save_o3d_visualize_fused_pointcloud(
        pcd_np, pcd_shared_np, output_path="tmp_fused.png",
        front=front, lookat=lookat, zoom=zoom, up=up,
    )


def test_convert_pointcloud_into_3D_video():
    # data_path = "./data/3DView/1/episode_00001/544_LIDAR"
    # json_meta_filepath = "data/3DView/1/episode_00001/measurements"

    data_path = "data/CoopernautVideoData/6_ Overtaking/CPT/2/259_LIDARFused"
    index_path = "data/CoopernautVideoData/6_ Overtaking/CPT/2/259_LIDARIndex"
    json_meta_filepath = "data/CoopernautVideoData/6_ Overtaking/CPT/2/measurements"

    output_path = data_path + "/3Dview/"
    Visualizer.convert_pointcloud_into_3D_video(
        data_path, output_path,
        index_path=index_path,
        front=front, lookat=lookat, zoom=zoom, up=up,
        json_meta_filepath=json_meta_filepath,
        fix_view_angle=True,
    )


def test_compile_video():
    result_dir = "test_traces/15/episode_00015/405_Left"
    Utils.compile_video(result_dir, "0_0_hud_info.mp4", [hud_files])

if __name__ == "__main__":
    # test_vis_point_cloud()
    # test_save_point_cloud()
    # test_save_shared_point_cloud()
    # test_convert_pointcloud_into_3D_video()

    test_compile_video()
