import copy

import numpy as np
import math
import time
from collections import defaultdict
from AVR import Utils
from AVR.DetectedObject import DetectedObject
from srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
import carla
import imageio

from MinkowskiEngine.utils import sparse_quantize
# from lidar_cython import fast_lidar
from numba import jit

from AVR.LidarProcessorConfig import LidarProcessorConfig


def extract_actor_bbox_ego_perspective(actor_list, th=0.2):
    object_bbox_list = []
    for d in actor_list:
        extent = d.bounding_box.extent

        pc = np.array([[-1 * (extent.y + th), -1 * (extent.x + th), extent.z],
                       [-1 * (extent.y + th), (extent.x + th), extent.z],
                       [(extent.y + th), (extent.x + th), extent.z],
                       [(extent.y + th), -1 * (extent.x + th), extent.z]])
        pc = np.array(
            Utils.transform_pointcloud(pc, d.get_transform(), CarlaActorPool.get_hero_actor().get_transform()))
        object_bbox_list.append(pc)
    return object_bbox_list


def get_min_corner_distance(va, vb):
    bbox_list_ego_perspective = extract_actor_bbox_ego_perspective([va, vb], th=0)
    box_a = bbox_list_ego_perspective[0]
    box_b = bbox_list_ego_perspective[1]
    min_dist = None
    for corner_a in box_a:
        for corner_b in box_b:
            dist = math.hypot(corner_a[0] - corner_b[0], corner_a[1] - corner_b[1])
            if min_dist is None or dist < min_dist:
                min_dist = dist
    return min_dist


# TODO move to elsewhere
def ComputeCoverage(fused_points):
    ## Compute Coverage for every 10cm x 10cm block
    coverage_count = 0
    coverage_matrix = {}
    for point in fused_points:
        x = 0
        y = 0
        if point.shape[0] == 1:
            point = np.array(point)[0]
        x = point[0] * 10
        y = point[1] * 10
        x = int(x)
        y = int(y)
        if x not in coverage_matrix:
            coverage_matrix[x] = [y]
            coverage_count += 1
        else:
            if y not in coverage_matrix[x]:
                coverage_matrix[x].append(y)
                coverage_count += 1
    return coverage_count


class PointCloudFrame(object):
    def __init__(self, id, pc, trans, frameId):
        self.id = id
        self.pc = pc
        self.trans = trans
        self.frameId = frameId


class LidarProcessResult(object):
    def __init__(self,
                 # occupancy_grid,
                 # grid_height_max,
                 # grid_height_min,
                 obstacle_grid,
                 obstacle_grid_with_margin_for_planning, filtered_actor_grid,
                 # drivable_lanytype_grid,
                 detected_object_list):
        # self.occupancy_grid = occupancy_grid
        # self.grid_height_max = grid_height_max
        # self.grid_height_min = grid_height_min
        self.obstacle_grid = obstacle_grid
        self.obstacle_grid_with_margin_for_planning = obstacle_grid_with_margin_for_planning
        self.filtered_actor_grid = filtered_actor_grid
        # self.drivable_lanytype_grid = drivable_lanytype_grid
        self.detected_object_list = detected_object_list


class LidarProcessResult_Fast(object):
    def __init__(self, grid_height_min, obstacle_grid, obstacle_grid_with_margin_for_planning, filtered_actor_grid,
                 # drivable_lanytype_grid,
                 detected_object_list):
        # self.occupancy_grid = occupancy_grid
        # self.grid_height_max = grid_height_max
        self.grid_height_min = grid_height_min
        self.obstacle_grid = obstacle_grid
        self.obstacle_grid_with_margin_for_planning = obstacle_grid_with_margin_for_planning
        self.filtered_actor_grid = filtered_actor_grid
        # self.drivable_lanytype_grid = drivable_lanytype_grid
        self.detected_object_list = detected_object_list


class LidarPreprocessor(object):
    """
    Static Class for lidar processing, to save from instantiation
        For sensor placing, (lidar coords)
            'z' represents the vertical axis (height),
            'y' represents the forward-backward axis,
            'x' represents the left-right axis.
        For vehicle coords
            x is forward backward
            y is left right
    """
    X_max = LidarProcessorConfig.X_max
    X_min = LidarProcessorConfig.X_min
    Y_max = LidarProcessorConfig.Y_max
    Y_min = LidarProcessorConfig.Y_min
    Z_max = LidarProcessorConfig.Z_max
    Z_min = LidarProcessorConfig.Z_min
    dX = LidarProcessorConfig.dX
    dY = LidarProcessorConfig.dY
    dZ = LidarProcessorConfig.dZ

    X_SIZE = LidarProcessorConfig.X_SIZE
    Y_SIZE = LidarProcessorConfig.Y_SIZE
    Z_SIZE = LidarProcessorConfig.Z_SIZE

    lidar_dim = [X_SIZE, Y_SIZE, Z_SIZE]
    lidar_depth_dim = [X_SIZE, Y_SIZE, 1]

    debug = False
    # depth_map_max = np.zeros((X_SIZE, Y_SIZE, 1)) - 10
    # depth_map_min = defaultdict(dict)
    # occupancy_grid = defaultdict(dict)

    x_bins = np.linspace(X_min, X_max + 1, X_SIZE + 1)
    y_bins = np.linspace(Y_min, Y_max + 1, Y_SIZE + 1)
    z_bins = np.linspace(Z_min, Z_max + 1, Z_SIZE + 1)

    @staticmethod
    def getX_grid(x):
        x_grid = int((x - LidarPreprocessor.X_min) / LidarPreprocessor.dX)
        return x_grid if 0 <= x_grid < LidarPreprocessor.X_SIZE else None

    @staticmethod
    def getY_grid(y):
        y_grid = int((y - LidarPreprocessor.Y_min) / LidarPreprocessor.dY)
        return y_grid if 0 <= y_grid < LidarPreprocessor.Y_SIZE else None

    @staticmethod
    def getZ_grid(z):
        z_grid = int((z - LidarPreprocessor.Z_min) / LidarPreprocessor.dZ)
        # if not 0 <= z_grid < LidarPreprocessor.Z_SIZE:
        #     print(z,z_grid)
        return z_grid if 0 <= z_grid < LidarPreprocessor.Z_SIZE else None

    @staticmethod
    def getX_meter(x):
        return x * LidarPreprocessor.dX + LidarPreprocessor.X_min

    @staticmethod
    def getY_meter(y):
        return y * LidarPreprocessor.dY + LidarPreprocessor.Y_min

    @staticmethod
    def getZ_meter(z):
        return z * LidarPreprocessor.dZ + LidarPreprocessor.Z_min

    @staticmethod
    def is_in_BEV(x, y, z):
        if x >= LidarPreprocessor.X_max or x <= LidarPreprocessor.X_min \
                or y >= LidarPreprocessor.Y_max or y <= LidarPreprocessor.Y_min \
                or z >= LidarPreprocessor.Z_max or z <= LidarPreprocessor.Z_min:
            return False
        return True

    @staticmethod
    def is_in_BEV_grid(x, y, z):
        if x >= LidarPreprocessor.X_SIZE or x < 0 \
                or y >= LidarPreprocessor.Y_SIZE or y < 0 \
                or z >= LidarPreprocessor.Z_SIZE or z < 0:
            return False
        return True

    # @staticmethod
    # def analyze_visibility(lidar, ego_transform, actor_list=None):
    #
    #     lidar_data = LidarPreprocessor.quantization(lidar, LidarPreprocessor.dX / Utils.QuantizationGrid,
    #                                                 LidarPreprocessor.dY / Utils.QuantizationGrid)
    #     actorId_points_dict = dict()
    #
    #     if actor_list is None:
    #         actor_list = CarlaDataProvider.get_world().get_actors()
    #     lidar_list = lidar_data.tolist()
    #     for i in range(len(lidar_list)):
    #         if len(lidar_list[i]) == 4:
    #             x, y, z, w = lidar_list[i]
    #         if len(lidar_list[i]) == 3:
    #             x, y, z = lidar_list[i]
    #
    #         for other_actor in actor_list:
    #             another_position = other_actor.get_transform().location
    #             # other_actor.get_transform().distance(hero_actor.get_transform())
    #             other_actor_position = [another_position.x, another_position.y, another_position.z]
    #             other_actor_position_numpy_array = np.array([other_actor_position])
    #             other_actor_position_ego_perspective = \
    #             Utils.world_to_car_transform(other_actor_position_numpy_array, ego_transform)[0]
    #             # just find a compound box for now, rotation can make it bigger

    @staticmethod
    def process_lidar(lidar, ego_transform, z_threshold, ego_actor, ego_length=0, ego_width=0):
        # print("Lidar data shape {}".format(lidar_data.shape))
        # print("Length {}, Width {}".format(ego_length, ego_width))

        time_start = time.time()
        """speed up"""
        # lidar_data = np.asarray(lidar)
        lidar_data = LidarPreprocessor.quantization(lidar, LidarPreprocessor.dX / Utils.QuantizationGrid,
                                                    LidarPreprocessor.dY / Utils.QuantizationGrid)
        # lidar_data = lidar_data[lidar_data[:, 2] < z_threshold]
        lidar_data = LidarPreprocessor.filter_points(lidar_data, 2, z_threshold, lower=False)

        time_quantize = time.time()
        if Utils.TIMEPROFILE: print("\t\t\t\tQuantization: {} s".format(time_quantize - time_start))

        obstacle_grid = None
        obstacle_grid_with_margin = None

        if Utils.EmulationMode:
            [occupancy_grid, depth_map_max, depth_map_min] = LidarPreprocessor.Lidar2BEV(lidar_data, ego_length,
                                                                                         ego_width)

            time_bev = time.time()
            if Utils.TIMEPROFILE: print("\t\t\t\tLidar2BEV: {} s".format(time_bev - time_quantize))

            z_threshold_mask, obstacle_grid_with_margin, obstacle_grid = LidarPreprocessor.filter_obstacle_grids_v2(
                depth_map_min, ego_length, ego_width, z_threshold)

            time_occupancy_grid = time.time()
            if Utils.TIMEPROFILE: print("\t\t\t\tObstacle Grid: {} s".format(time_occupancy_grid - time_bev))

            detected_object_list, filtered_actor_grid = LidarPreprocessor.isolated_object_detection_v2(
                occupancy_grid,
                depth_map_min,
                z_threshold,
                ego_transform,
                grid_to_search=obstacle_grid)  # care only about objects on drivable space
            time_obj = time.time()
            if Utils.TIMEPROFILE: print("\t\t\t\tIsolated Island: {} s".format(time_obj - time_occupancy_grid))

        else:
            detected_object_list, filtered_actor_grid, depth_map_min = LidarPreprocessor.get_objects_from_carla(
                lidar_data,
                z_threshold,
                ego_actor)

            z_threshold_mask, obstacle_grid_with_margin, obstacle_grid = LidarPreprocessor.filter_obstacle_grids_v2(
                depth_map_min, ego_length, ego_width, z_threshold)

            time_obj = time.time()
            if Utils.TIMEPROFILE: print("\t\t\t\tGet Objects: {} s".format(time_obj - time_quantize))

        # if Utils.TIMEPROFILE: print("\t\t\tPerception Total: {} s".format(time_obj - time_start))
        # print(np.sum(obstacle_grid != 0))
        result = LidarProcessResult(
            # occupancy_grid=occupancy_grid,
            # grid_height_max=depth_map_max,
            # grid_height_min=depth_map_min,
            obstacle_grid=obstacle_grid,
            obstacle_grid_with_margin_for_planning=obstacle_grid_with_margin,
            filtered_actor_grid=filtered_actor_grid,
            # drivable_lanytype_grid=drivable_lanytype_grid,
            detected_object_list=detected_object_list)
        return result

    @staticmethod
    def process_lidar_fast(lidar_data, ego_transform, z_threshold, ego_length=0, ego_width=0):
        # print("Lidar data shape {}".format(lidar_data.shape))
        # print("Length {}, Width {}".format(ego_length, ego_width))
        time_start = time.time()
        # [occupancy_grid, depth_map_max, depth_map_min] = LidarPreprocessor.Lidar2BEV(lidar_data, ego_length, ego_width)
        # [occupancy_grid, depth_map_max, depth_map_min] = LidarPreprocessor.Lidar2BEV_Downsampled(lidar_data, ego_length, ego_width)
        depth_map_min, voxels = LidarPreprocessor.Lidar2BEV_Fast(lidar_data)

        time_bev = time.time()
        # if Utils.TIMEPROFILE: print("Lidar2BEV: {} s".format(time_bev - time_start))
        z_threshold_mask, obstacle_grid_with_margin, obstacle_grid = LidarPreprocessor.filter_obstacle_grids(
            depth_map_min, ego_length, ego_width, z_threshold)
        time_occupancy_grid = time.time()
        # if Utils.TIMEPROFILE: print("Obstacle Grid: {} s".format(time_occupancy_grid - time_bev))
        detected_object_list, filtered_actor_grid = LidarPreprocessor.isolated_object_detection_fast(voxels,
                                                                                                     depth_map_min,
                                                                                                     z_threshold,
                                                                                                     ego_transform,
                                                                                                     grid_to_search=obstacle_grid)  # care only about objects on drivable space
        time_obj = time.time()
        # if Utils.TIMEPROFILE: print("Isolated Island: {} s".format(time_obj - time_occupancy_grid))
        # print(np.sum(obstacle_grid != 0))
        result = LidarProcessResult_Fast(
            grid_height_min=depth_map_min, obstacle_grid=obstacle_grid,
            obstacle_grid_with_margin_for_planning=obstacle_grid_with_margin,
            filtered_actor_grid=filtered_actor_grid,
            # drivable_lanytype_grid=drivable_lanytype_grid,
            detected_object_list=detected_object_list)
        return result

    @staticmethod
    def is_self_lidar_data_vectorize(x, y, length, width):
        margin = 0.1  # 10 cm
        """compensate for lidar setup location"""
        compensated_y = y - Utils.LidarRoofForwardDistance
        return ((length / 2.0 + margin) >= compensated_y) & (compensated_y >= (-1 * length / 2.0 - margin)) & (
                (width / 2.0 + margin) >= x) & (x >= (
                -1 * width / 2.0 - margin))

    @staticmethod
    def is_self_lidar_data(x, y, length, width):
        margin = 0.1  # 10 cm
        x, y = float(x), float(y)
        """compensate for lidar setup location"""
        compensated_y = y - Utils.LidarRoofForwardDistance
        if (length / 2.0 + margin) >= compensated_y >= (-1 * length / 2.0 - margin) and (width / 2.0 + margin) >= x >= (
                -1 * width / 2.0 - margin):
            return True
        return False

    @staticmethod
    def is_self_grid(i, j, ego_length, ego_width):
        margin = 0  # 0.5 m
        length_grid = math.ceil(ego_length / LidarPreprocessor.dY / 2)
        width_grid = math.ceil(ego_width / LidarPreprocessor.dX / 2)
        """compensate for lidar setup location"""
        compensated_j = j - int(Utils.LidarRoofForwardDistance / LidarPreprocessor.dY)
        if LidarPreprocessor.Y_SIZE / 2 - length_grid - margin <= compensated_j <= LidarPreprocessor.Y_SIZE / 2 + length_grid + margin \
                and LidarPreprocessor.X_SIZE / 2 - width_grid - margin <= i <= LidarPreprocessor.X_SIZE / 2 + width_grid + margin:
            return True
        return False

    # TODO: replace with v2 version

    @staticmethod
    def Lidar_Voxelization(lidar, step=5.0):
        """
        quantize using a fine grid
        return: LidarProcessor dX dY grids
        """
        dx, dy = LidarPreprocessor.dX / step, LidarPreprocessor.dY / step
        lidar = np.asarray(lidar)
        coord = lidar[:, :2]
        coord, downsampled_lidar = sparse_quantize(coord, lidar, quantization_size=(dx, dy))
        """ shift to -70,-70 as 0,0 """
        coord += [int(LidarPreprocessor.X_SIZE / 2 * step), int(LidarPreprocessor.Y_SIZE / 2 * step)]
        coord = np.floor_divide(coord, step).astype('int32')
        return coord, downsampled_lidar

    @staticmethod
    def Sparse_Quantize(lidar):
        """
        quantize the lidar into sparse points
        """
        scale = 1.0
        dx, dy, dz = LidarPreprocessor.dX, LidarPreprocessor.dY, LidarPreprocessor.dZ

        lidar = lidar[:, :3]
        Z_max = -0.3
        Z_min = -2.5
        lidar = LidarPreprocessor.filter_lidar_by_boundary(lidar, LidarPreprocessor.X_max,
                                                           LidarPreprocessor.X_min,
                                                           LidarPreprocessor.Y_max,
                                                           LidarPreprocessor.Y_min,
                                                           Z_max,
                                                           Z_min)
        lidar = sparse_quantize(lidar, quantization_size=(dx * scale, dy * scale, dz * scale))
        lidar = np.array(lidar, dtype=np.float)
        lidar[:, 0] = lidar[:, 0] * dx * scale
        lidar[:, 1] = lidar[:, 1] * dy * scale
        lidar[:, 2] = lidar[:, 2] * dz * scale
        lidar = np.array(lidar)

        return lidar

    # @staticmethod
    # def Lidar2BEV_Downsampled(lidar_data, ego_length=0, ego_width=0):
    #     # print(lidar_data.shape)
    #     # start = time.time()
    #     coords, downsampled_lidar = LidarPreprocessor.Lidar_Voxelization(lidar_data)
    #     # vox = time.time()
    #     occupancy_grid = defaultdict(dict)
    #     depth_map_min = np.zeros((LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE)) + 10
    #     depth_map_max = np.zeros((LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE)) - 10
    #     downsampled_lidar = downsampled_lidar.tolist()
    #     # tolist = time.time()
    #     for i in range(len(coords)):
    #         x_m, y_m, z_m = downsampled_lidar[i]
    #         if LidarPreprocessor.is_self_lidar_data(x_m, y_m, ego_length, ego_width):
    #             continue
    #
    #         x, y = coords[i]
    #         if x in occupancy_grid.keys() and y in occupancy_grid[x].keys():
    #             occupancy_grid[x][y] += [[x_m, y_m, z_m]]
    #         else:
    #             occupancy_grid[x][y] = [[x_m, y_m, z_m]]
    #
    #     for x in occupancy_grid:
    #         for y in occupancy_grid[x]:
    #             depth_map_max[x][y] = max(np.array(occupancy_grid[x][y])[:, 2])
    #             depth_map_min[x][y] = min(np.array(occupancy_grid[x][y])[:, 2])
    #
    #     # end = time.time()
    #     # print("Lidar2BEV_Downsampled: {}(Vox:{}+tolist:{}+forloop:{}). {}".format(end - start, vox-start, tolist-vox, end-tolist, lidar_data.shape))
    #     return occupancy_grid, depth_map_max, depth_map_min

    @staticmethod
    def Lidar2BEV_ME(lidar):
        lidar = np.asarray(lidar)
        coord, feat = lidar[:, :2], lidar
        coord, feat = sparse_quantize(coord, feat, quantization_size=(LidarPreprocessor.dX, LidarPreprocessor.dY))
        """ shift to -70,-70 as 0,0 """
        coord += [140, 140]
        return coord, feat

    @staticmethod
    def quantization(lidar, dx, dy):
        lidar = np.asarray(lidar)
        coord, feat = lidar[:, :2], lidar
        coord, feat = sparse_quantize(coord, feat, quantization_size=(dx, dy))
        return feat

    @staticmethod
    def Lidar2BEV_Fast(lidar_data, ego_length=0, ego_width=0):
        # print(lidar_data.shape)
        # start = time.time()
        coords, point = LidarPreprocessor.Lidar2BEV_ME(lidar_data)
        # vox = time.time()
        depth_map_min = np.zeros((LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE)) + 10
        # print(coords.shape)
        # print(depth_map_min[coords[:,0], coords[:,1]].shape)
        # print(point.shape)
        # print(point[:, 2].shape)
        depth_map_min[coords[:, 0], coords[:, 1]] = point[:, 2]
        # depth_map_min = np.squeeze(depth_map_min)
        # end = time.time()
        # print(
        #     "Lidar2BEV_Fast: {}(Vox:{}+depthmap:{}). {}".format(end - start, vox - start, end - vox, lidar_data.shape))
        return depth_map_min, point

    @staticmethod
    @jit(nopython=True)
    def filter_lidar_by_boundary(lidar_data, X_max, X_min, Y_max, Y_min, Z_max, Z_min):
        lidar_data = lidar_data[(lidar_data[..., 0] < X_max)
                                & (lidar_data[..., 0] > X_min)
                                & (lidar_data[..., 1] < Y_max)
                                & (lidar_data[..., 1] > Y_min)
                                & (lidar_data[..., 2] <= Z_max)
                                & (lidar_data[..., 2] >= Z_min)]
        return lidar_data

    @staticmethod
    @jit(nopython=True)
    def filter_self_lidar(lidar_data, length, width, Y_compensation):
        """compensate for lidar setup location"""

        margin = 0.1  # 10 cm
        lidar_data = lidar_data[(lidar_data[..., 1] > (length / 2.0 + margin) + Y_compensation)
                                | (lidar_data[..., 1] < (-1 * length / 2.0 - margin) + Y_compensation)
                                | (lidar_data[..., 0] > (width / 2.0 + margin))
                                | (lidar_data[..., 0] < (-1 * width / 2.0 - margin))]

        return lidar_data

    @staticmethod
    def filter_object_points(lidar, grid_x, grid_y):

        x_min = LidarPreprocessor.getX_meter(grid_x)
        x_max = LidarPreprocessor.getX_meter(grid_x + 1)
        y_min = LidarPreprocessor.getX_meter(grid_y)
        y_max = LidarPreprocessor.getX_meter(grid_y + 1)
        # print(lidar_data.shape)
        # print((np.squeeze(lidar_data[:, 0] < x_max)).shape)

        ret = LidarPreprocessor.filter_object_points_by_bbox(lidar, x_min, x_max, y_min, y_max)
        # print("Filtering lidar for gird ({},{}): {} points".format(grid_x, grid_y, len(ret)))
        return ret

    @staticmethod
    def filter_object_points_by_bbox(lidar, x_min, x_max, y_min, y_max):
        ret = copy.deepcopy(lidar)
        ret = LidarPreprocessor.filter_points(ret, 0, x_min, lower=True)
        ret = LidarPreprocessor.filter_points(ret, 0, x_max, lower=False)
        ret = LidarPreprocessor.filter_points(ret, 1, y_min, lower=True)
        ret = LidarPreprocessor.filter_points(ret, 1, y_max, lower=False)
        return ret

    @staticmethod
    @jit(nopython=True)
    def filter_points(lidar_data, axis, threshold, lower):

        # if len(lidar_data) == 0: return lidar_data
        # if len(lidar_data) == 1:
        #     if lower:
        #         if lidar_data[0, axis] > threshold:
        #             return lidar_data
        #     else:
        #         if lidar_data[0, axis] < threshold:
        #             return lidar_data
        #     return None

        # assert len(lidar_data.shape) == 2 and lidar_data.shape[1] == 3, print(lidar_data.shape)

        if lower:
            filter = lidar_data[:, axis] > threshold
        else:
            filter = lidar_data[:, axis] < threshold

        # try:
        # filter = filter.reshape((filter.shape[0], 1))
        # if len(filter.shape) > 1:
        #     print("Before shaping{}".format(filter.shape))
        #     filter = np.squeeze(filter, axis=1)
        #     # filter = filter.reshape((filter.shape[0],))
        #     print("After shaping {}".format(filter.shape))
        # if len(filter.shape) > 1:
        #     filter = filter[0, :]
        #     # filter = np.squeeze(filter, axis=0)
        #     print("After selection {}".format(filter.shape))
        #     """doesn't work for large matrix.... numpy bug"""

        # assert len(filter.shape) == 1, print(filter.shape)
        ret = lidar_data[filter, :]
        # assert len(ret.shape) == 2 and ret.shape[1] == 3, print(
        #     "lidar:{}, filter:{}, ret:{}".format(lidar_data.shape, filter.shape, ret.shape))
        # except:
        #     raise Exception("Point filtering error, lidar shape {}, filter shape {}".format(lidar_data.shape, filter.shape))

        return ret

    @staticmethod
    def Lidar2BEV(lidar, ego_length=0, ego_width=0):
        """
            Convert lidar to bird eye view
            BEV format:
                3D occupancy grid (0 or 1),

                plus an average reflection value per 2D grid

                (not for now, carla has no reflection data)

        :param lidar_data:
        :return:
        """
        lidar_data = np.asarray(lidar)
        lidar_data = LidarPreprocessor.filter_lidar_by_boundary(lidar_data, LidarPreprocessor.X_max,
                                                                LidarPreprocessor.X_min, LidarPreprocessor.Y_max,
                                                                LidarPreprocessor.Y_min, LidarPreprocessor.Z_max,
                                                                LidarPreprocessor.Z_min)

        lidar_data = LidarPreprocessor.filter_self_lidar(lidar_data, ego_length, ego_width,
                                                         Utils.LidarRoofForwardDistance)

        lidar_list = lidar_data.tolist()
        occupancy_grid = dict()
        depth_map_min = np.zeros((LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE)) + 10
        depth_map_max = np.zeros((LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE)) - 10
        if len(lidar_list) > 0:
            occupancy_grid, depth_map_max, depth_map_min = LidarPreprocessor.get_occupancy_grid(lidar_list,
                                                                                                LidarPreprocessor.X_min,
                                                                                                LidarPreprocessor.dX,
                                                                                                LidarPreprocessor.Y_min,
                                                                                                LidarPreprocessor.dY,
                                                                                                LidarPreprocessor.X_SIZE,
                                                                                                LidarPreprocessor.Y_SIZE)
        return occupancy_grid, depth_map_max, depth_map_min

    @staticmethod
    # TODO: commented due to no support of defaultdict or 2d dictionary
    # @jit(nopython=True)
    def get_occupancy_grid(lidar_list, X_min, dX, Y_min, dY, X_SIZE, Y_SIZE):

        # occupancy_grid = dict()
        occupancy_grid = defaultdict(dict)
        depth_map_min = np.zeros((X_SIZE, Y_SIZE)) + 10
        depth_map_max = np.zeros((X_SIZE, Y_SIZE)) - 10
        # print("Lidar Points: {}".format(num_points))
        for i in range(len(lidar_list)):
            if len(lidar_list[i]) == 4:
                x, y, z, w = lidar_list[i]
            if len(lidar_list[i]) == 3:
                x, y, z = lidar_list[i]

            x_grid = int((x - X_min) / dX)
            y_grid = int((y - Y_min) / dY)

            if x_grid in occupancy_grid.keys() and y_grid in occupancy_grid[x_grid].keys():
                occupancy_grid[x_grid][y_grid] += [lidar_list[i][:3]]
            else:
                occupancy_grid[x_grid][y_grid] = [lidar_list[i][:3]]

            depth_map_max[x_grid][y_grid] = max(z, depth_map_max[x_grid][y_grid])
            depth_map_min[x_grid][y_grid] = min(z, depth_map_min[x_grid][y_grid])

        return occupancy_grid, depth_map_max, depth_map_min

    @staticmethod
    def Lidar2BEV_v2(points, ego_length=0, ego_width=0, max_per_pixel=1, customize_z=False):
        points = np.asarray(points)

        if customize_z:
            Z_max = -0.3
            Z_min = -2.5
        else:
            Z_max = LidarPreprocessor.Z_max
            Z_min = LidarPreprocessor.Z_min
        '''
        points = LidarPreprocessor.filter_lidar_by_boundary(points, LidarPreprocessor.X_max,
                                                                  LidarPreprocessor.X_min, 
                                                                  LidarPreprocessor.Y_max,
                                                                  LidarPreprocessor.Y_min, 
                                                                  Z_max,
                                                                  Z_min)
        '''
        points = points[
            (points[..., 0] < LidarPreprocessor.X_max)
            & (points[..., 0] > LidarPreprocessor.X_min)
            & (points[..., 1] < LidarPreprocessor.Y_max)
            & (points[..., 1] > LidarPreprocessor.Y_min)
            & (points[..., 2] <= Z_max)  # LidarPreprocessor.Z_max)
            & (points[..., 2] >= Z_min)  # LidarPreprocessor.Z_min)
            ]
        hist = \
            np.histogramdd(points, bins=(LidarPreprocessor.x_bins, LidarPreprocessor.y_bins, LidarPreprocessor.z_bins))[
                0]

        hist[hist > max_per_pixel] = max_per_pixel

        return hist

    @staticmethod
    def occupancy_grid_dict_to_numpy(occupancy_grid):
        BEV = np.zeros((LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE, LidarPreprocessor.Z_SIZE))
        for x in occupancy_grid:
            for y in occupancy_grid[x]:
                for point in occupancy_grid[x][y]:
                    z = LidarPreprocessor.getZ_grid(point[2])
                    if not LidarPreprocessor.is_in_BEV(point[0], point[1], point[2]):
                        #    print("Not in BEV")
                        continue
                    BEV[x, y, z] = 1
        return BEV

    @staticmethod
    def filter_obstacle_grids(depth_map_min, ego_length, ego_width, z_threshold):
        grid_obstacle_with_margin = np.zeros(shape=(LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE), dtype=int)
        grid_obstacle = np.zeros(shape=(LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE), dtype=int)
        z_threshold_mask = np.array(depth_map_min <= z_threshold)
        for i in range(z_threshold_mask.shape[0]):
            for j in range(z_threshold_mask.shape[1]):
                # clear ego, do not add margin... must do this, if not, later group assignment can clear the point, but won't clear margins
                if LidarPreprocessor.is_self_grid(i, j, ego_length, ego_width):
                    continue
                if z_threshold_mask[i, j]:
                    grid_obstacle[i, j] = 1
                    # LidarPreprocessor.group_assign_grid_value_by_dimension_block(grid_obstacle_with_margin, i, j, ego_length, ego_width, 1)
                    LidarPreprocessor.group_assign_grid_value_by_dimension_circle(grid_obstacle_with_margin, i, j,
                                                                                  (ego_length + ego_width) / 2 / 2, 1)

        source_x = LidarPreprocessor.getX_grid(0)
        # source_y = LidarPreprocessor.getY_grid(-3)
        # # clear drive path
        # LidarPreprocessor.group_assign_grid_value_by_dimension_block(grid_obstacle_with_margin, source_x,
        #                                                              source_y, ego_length,
        #                                                              ego_width, 0)
        # source_y = LidarPreprocessor.getY_grid(-ego_length/2)
        # LidarPreprocessor.group_assign_grid_value_by_dimension_block(grid_obstacle_with_margin, source_x,
        #                                                              source_y, ego_length,
        #                                                              ego_width, 0)
        # clear ego
        source_y = LidarPreprocessor.getY_grid(
            Utils.LidarRoofForwardDistance)  # the sensor is mounted 0.7 forward to center
        # LidarPreprocessor.group_assign_grid_value_by_dimension_circle(grid_obstacle_with_margin, source_x,
        #                                                              source_y, max(ego_length,ego_width)/2, 0)
        # LidarPreprocessor.group_assign_grid_value_by_dimension_circle(grid_obstacle, source_x,
        #                                                              source_y, max(ego_length,ego_width)/2, 0)
        LidarPreprocessor.group_assign_grid_value_by_dimension_block(grid_obstacle_with_margin, source_x,
                                                                     source_y, ego_length,
                                                                     ego_width, 0)
        LidarPreprocessor.group_assign_grid_value_by_dimension_block(grid_obstacle, source_x,
                                                                     source_y, ego_length,
                                                                     ego_width, 0)

        return z_threshold_mask, grid_obstacle_with_margin, grid_obstacle

    @staticmethod
    def filter_obstacle_grids_v2(depth_map_min, ego_length, ego_width, z_threshold):
        grid_obstacle_with_margin = np.zeros(shape=(LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE), dtype=int)
        grid_obstacle = np.zeros(shape=(LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE), dtype=int)
        z_threshold_mask = np.array(depth_map_min <= z_threshold)
        length_grid = math.ceil(ego_length / LidarPreprocessor.dY / 2)
        width_grid = math.ceil(ego_width / LidarPreprocessor.dX / 2)
        margin = 0
        center_X_grid = LidarPreprocessor.X_SIZE / 2
        center_Y_grid = LidarPreprocessor.Y_SIZE / 2
        compensate_Y_grid = int(Utils.LidarRoofForwardDistance / LidarPreprocessor.dY)
        radius = (ego_length + ego_width) / 2 / 2
        radius_grid = math.ceil(radius / min(LidarPreprocessor.dY, LidarPreprocessor.dX))

        grid_obstacle, grid_obstacle_with_margin = LidarPreprocessor.get_obstacle_grid_with_margin(grid_obstacle,
                                                                                                   grid_obstacle_with_margin,
                                                                                                   z_threshold_mask,
                                                                                                   compensate_Y_grid,
                                                                                                   radius_grid,
                                                                                                   margin,
                                                                                                   center_X_grid,
                                                                                                   center_Y_grid,
                                                                                                   length_grid,
                                                                                                   width_grid,
                                                                                                   LidarPreprocessor.X_SIZE,
                                                                                                   LidarPreprocessor.Y_SIZE)

        source_x = LidarPreprocessor.getX_grid(0)
        # clear ego
        source_y = LidarPreprocessor.getY_grid(
            Utils.LidarRoofForwardDistance)  # the sensor is mounted 0.7 forward to center

        LidarPreprocessor.group_assign_grid_value_by_dimension_block(grid_obstacle_with_margin, source_x,
                                                                     source_y, ego_length,
                                                                     ego_width, 0)
        LidarPreprocessor.group_assign_grid_value_by_dimension_block(grid_obstacle, source_x,
                                                                     source_y, ego_length,
                                                                     ego_width, 0)

        return z_threshold_mask, grid_obstacle_with_margin, grid_obstacle

    @staticmethod
    @jit(nopython=True)
    def get_obstacle_grid_with_margin(grid_obstacle, grid_obstacle_with_margin, z_threshold_mask, compensate_Y_grid,
                                      radius_grid, margin, center_X_grid, center_Y_grid, length_grid, width_grid,
                                      X_SIZE, Y_SIZE):
        offset = 0
        for i in range(z_threshold_mask.shape[0]):
            for j in range(z_threshold_mask.shape[1]):
                # clear ego, do not add margin... must do this, if not, later group assignment can clear the point, but won't clear margins
                """compensate for lidar setup location"""
                compensated_j = j - compensate_Y_grid
                if center_Y_grid - length_grid - margin <= compensated_j <= center_Y_grid + length_grid + margin \
                        and center_X_grid - width_grid - margin <= i <= center_X_grid + width_grid + margin:
                    continue

                if z_threshold_mask[i, j]:
                    grid_obstacle[i, j] = 1
                    x_min = int((max(0, i - radius_grid - offset)))
                    x_max = int((min(X_SIZE - 1, i + radius_grid + offset)))
                    y_min = int((max(0, j - radius_grid - offset)))
                    y_max = int((min(Y_SIZE - 1, j + radius_grid + offset)))
                    for x in range(x_min, x_max + 1):
                        for y in range(y_min, y_max + 1):
                            if math.hypot(x - i, y - j) < radius_grid + offset:
                                # print("({},{})={} < {}".format(x-i,y-j, math.hypot(x-i,y-j), radius_grid))
                                grid_obstacle_with_margin[x][y] = 1
        return grid_obstacle, grid_obstacle_with_margin

    @staticmethod
    def group_assign_grid_value_by_dimension_circle(grid_obstacle, i, j, radius, value):
        offset = 0
        radius_grid = math.ceil(radius / min(LidarPreprocessor.dY, LidarPreprocessor.dX)) + 1
        x_min = int((max(0, i - radius_grid - offset)))
        x_max = int((min(LidarPreprocessor.X_SIZE - 1, i + radius_grid + offset)))
        y_min = int((max(0, j - radius_grid - offset)))
        y_max = int((min(LidarPreprocessor.Y_SIZE - 1, j + radius_grid + offset)))
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if math.hypot(x - i, y - j) <= radius_grid + offset:
                    # print("({},{})={} < {}".format(x-i,y-j, math.hypot(x-i,y-j), radius_grid))
                    grid_obstacle[x][y] = value

    @staticmethod
    def group_assign_grid_value_by_dimension_block(grid_obstacle, i, j, length, width, value):
        offset = 0
        length_grid = math.ceil(length / LidarPreprocessor.dY / 2)
        width_grid = math.ceil(width / LidarPreprocessor.dX / 2)

        x_min = int((max(0, i - width_grid - offset)))
        x_max = int((min(LidarPreprocessor.X_SIZE - 1, i + width_grid + offset)))
        y_min = int((max(0, j - length_grid - offset)))
        y_max = int((min(LidarPreprocessor.Y_SIZE - 1, j + length_grid + offset)))

        grid_obstacle = LidarPreprocessor.group_assign(grid_obstacle, value, x_min, x_max, y_min, y_max)

    @staticmethod
    @jit(nopython=True)
    def group_assign(grid_obstacle, value, x_min, x_max, y_min, y_max):
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                grid_obstacle[x][y] = value
        return grid_obstacle

    # @staticmethod
    # def isolated_object_detection_cython(occupancy_grid, depth_map_min, z_threshold, ego_Trans, grid_to_search=None):
    #     if grid_to_search is None:
    #         grids_to_search = np.array(depth_map_min <= z_threshold)
    #     else:
    #         grids_to_search = np.array(grid_to_search)  # cuz this function mutate the drivable grid as check marks
    #
    #     detected_object_list, filtered_actor_grid = fast_lidar.isolated_object_detection_cython(LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE, occupancy_grid, ego_Trans, z_threshold, depth_map_min, grids_to_search)
    #     return detected_object_list, filtered_actor_grid

    @staticmethod
    def isolated_object_detection(occupancy_grid, depth_map_min, z_threshold, ego_Trans, grid_to_search=None):
        filtered_actor_grid = np.zeros(shape=(LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE), dtype=int)
        if grid_to_search is None:
            grids_to_search = np.array(depth_map_min <= z_threshold)
        else:
            grids_to_search = np.array(grid_to_search)  # cuz this function mutate the drivable grid as check marks

        # number of isolated islands
        ans = 0
        detected_object_list = []
        # isolation_margin = 1

        # print(mvs)
        mvs = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        # print(grid_to_search.shape)
        for i in range(grids_to_search.shape[0]):
            for j in range(grids_to_search.shape[1]):
                # print(grids_to_search[i, j])
                if not grids_to_search[i, j]:
                    continue
                # TODO: uncomment if carla lane type semantic becomes better
                drivable, _ = LidarPreprocessor.is_valid_grid_lanetype(i, j, ego_Trans,
                                                                       valid_lane_type=Utils.perception_lane_type)
                if not drivable:
                    continue
                stack = [[i, j]]
                detected_object = DetectedObject(ans)
                while (stack):
                    pos = stack.pop()
                    grids_to_search[pos[0], pos[1]] = 0
                    if (pos[0] not in occupancy_grid) or (pos[1] not in occupancy_grid[pos[0]]):
                        continue

                    # dist = math.hypot((pos[0] - LidarPreprocessor.X_SIZE / 2) * LidarPreprocessor.dX,
                    #                   (pos[1] - LidarPreprocessor.Y_SIZE / 2) * LidarPreprocessor.dY)
                    # if dist > 30:
                    #     continue

                    detected_object.insert_occupancy_grid(pos[0], pos[1], depth_map_min[pos[0]][pos[1]])
                    detected_object.insert_point_cloud(occupancy_grid[pos[0]][pos[1]])

                    # dynamic search margin
                    # isolation_margin = 0.00375 * dist * dist
                    # isolation_margin /= LidarPreprocessor.dX
                    # isolation_margin = math.ceil(isolation_margin)
                    # isolation_margin = np.clip(isolation_margin, 1, 3)
                    # # print("pos:{},dist:{},margin{}".format(pos, dist, isolation_margin))
                    #
                    # mvs = []
                    # for i in range(-isolation_margin, isolation_margin + 1):
                    #     for j in range(-isolation_margin, isolation_margin + 1):
                    #         if i == 0 and j == 0:
                    #             continue
                    #         mvs.append([i, j])
                    for mv in mvs:
                        x, y = pos[0] + mv[0], pos[1] + mv[1]
                        if 0 <= x < grids_to_search.shape[0] and 0 <= y < grids_to_search.shape[1] and grids_to_search[
                            x, y]:
                            stack.append([x, y])
                """ filter by area and point clouds"""
                if len(detected_object.occupancy_grid_list) > 2 or len(detected_object.point_cloud_list) > 20:
                    ans += 1
                    detected_object.get_bounding_box(z_threshold)
                    detected_object_list.append(detected_object)
                    for [x, y, z] in detected_object.occupancy_grid_list:
                        filtered_actor_grid[x, y] = 1
                else:
                    del detected_object
        return detected_object_list, filtered_actor_grid

    @staticmethod
    @jit(nopython=True)
    def isolated_islands(grids_to_search):
        """
        Return a list of objects' grids, each object grid is a list of [x,y]
        """
        mvs = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        # print(grid_to_search.shape)
        obj_grid_list = [[[-1, -1]]]
        for i in range(grids_to_search.shape[0]):
            for j in range(grids_to_search.shape[1]):
                # print(grids_to_search[i, j])
                if not grids_to_search[i, j]:
                    continue
                stack = [[i, j]]
                obj_grid = [[i, j]]
                while (stack):
                    pos = stack.pop()
                    grids_to_search[pos[0], pos[1]] = 0
                    for mv in mvs:
                        x, y = pos[0] + mv[0], pos[1] + mv[1]
                        if 0 <= x < grids_to_search.shape[0] and 0 <= y < grids_to_search.shape[1] and grids_to_search[
                            x, y]:
                            stack.append([x, y])
                            obj_grid.append([x, y])
                obj_grid_list.append(obj_grid)
        return obj_grid_list[1:]

    @staticmethod
    def get_objects_from_carla(lidar, z_threshold, ego_actor, actor_list=None):
        """
        This is a speed up alternative for object detection, if users are only interested in planning related experiments
        """
        if actor_list is None:
            actor_list = CarlaDataProvider.get_world().get_actors()
            actor_list = actor_list.filter("*vehicle*")

        obj_id = 0
        detected_object_list = []
        filtered_actor_grid = np.zeros(shape=(LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE), dtype=int)
        depth_map_min = np.zeros(shape=(LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE)) + 10

        for other_actor in actor_list:
            if other_actor.id == ego_actor.id:
                continue
            another_position = other_actor.get_transform().location
            other_actor_position = [another_position.x, another_position.y, another_position.z]
            other_actor_position_numpy_array = np.array([other_actor_position])
            other_actor_position_ego_perspective = \
                Utils.world_to_car_transform(other_actor_position_numpy_array, ego_actor.get_transform())[0]

            x = other_actor_position_ego_perspective[0]
            y = other_actor_position_ego_perspective[1]
            z = other_actor_position_ego_perspective[2]

            grid_x = LidarPreprocessor.getX_grid(x)
            grid_y = LidarPreprocessor.getY_grid(y)
            grid_z = LidarPreprocessor.getZ_grid(z)

            # expand by 2 grid each dim
            # grid_list = []
            if grid_x is None or grid_y is None or grid_z is None:
                continue

            x = LidarPreprocessor.getX_meter(grid_x)
            y = LidarPreprocessor.getY_meter(grid_y)
            detected_object = DetectedObject(obj_id)
            dummy_grid_range = 10
            for i in range(-dummy_grid_range, dummy_grid_range):
                for j in range(-dummy_grid_range, dummy_grid_range):
                    if LidarPreprocessor.is_in_BEV_grid(grid_x + i, grid_y + j, grid_z):

                        subset_lidar = LidarPreprocessor.filter_lidar_by_boundary(
                            lidar,
                            X_max=x + (i+1) * LidarPreprocessor.dX, X_min=x + i * LidarPreprocessor.dX,
                            Y_max=y + (j+1) * LidarPreprocessor.dY, Y_min=y + j * LidarPreprocessor.dY,
                            Z_max=LidarPreprocessor.Z_max, Z_min=LidarPreprocessor.Z_min)

                        # check height for this grid
                        if not np.size(subset_lidar):
                            continue
                        # print(subset_lidar)
                        z_min = np.min(subset_lidar[:, 2])
                        depth_map_min[grid_x+i][grid_y+j] = z_min
                        # print(z_min, z_threshold)
                        if z_min <= z_threshold:
                            detected_object.insert_occupancy_grid(grid_x + i, grid_y + j, z_min)
                            subset_lidar = subset_lidar.tolist()
                            detected_object.insert_point_cloud(subset_lidar)

            obj_id += 1
            detected_object.get_bounding_box(z_threshold)
            # detected_object.print()
            detected_object_list.append(detected_object)
            for [x, y, z] in detected_object.occupancy_grid_list:
                filtered_actor_grid[x, y] = 1

        return detected_object_list, filtered_actor_grid, depth_map_min

    @staticmethod
    def isolated_object_detection_v2(occupancy_grid, depth_map_min, z_threshold, ego_Trans, grid_to_search=None):
        filtered_actor_grid = np.zeros(shape=(LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE), dtype=int)
        if grid_to_search is None:
            grids_to_search = np.array(depth_map_min <= z_threshold)
        else:
            grids_to_search = np.array(grid_to_search)  # cuz this function mutate the drivable grid as check marks

        obj_grid_list = LidarPreprocessor.isolated_islands(grids_to_search)

        # number of isolated islands
        obj_id = 0
        detected_object_list = []

        for obj_grid in obj_grid_list:

            detected_object = DetectedObject(obj_id)
            # dontcare = True
            for idx in range(len(obj_grid)):
                [i, j] = obj_grid[idx]
                if idx == 0:
                    """objects lane type that we should care"""
                    # drivable, _ = LidarPreprocessor.is_drivable_grid(i, j, ego_Trans, valid_lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk)
                    care, _ = LidarPreprocessor.is_valid_grid_lanetype(i, j, ego_Trans,
                                                                       valid_lane_type=Utils.perception_lane_type,
                                                                       debug=False)
                    if not care:
                        break

                if (i not in occupancy_grid) or (j not in occupancy_grid[i]):
                    continue
                detected_object.insert_occupancy_grid(i, j, depth_map_min[i][j])
                detected_object.insert_point_cloud(occupancy_grid[i][j])

            """ filter by area and point clouds"""
            if len(detected_object.point_cloud_list) == 0:
                # print("Obj is empty, should not reach here")
                # TODO: investigate why there is 0 point objects detected
                del detected_object
            elif len(detected_object.occupancy_grid_list) > 2 or len(detected_object.point_cloud_list) > 20:
                obj_id += 1
                detected_object.get_bounding_box(z_threshold)
                # detected_object.print()
                detected_object_list.append(detected_object)
                for [x, y, z] in detected_object.occupancy_grid_list:
                    filtered_actor_grid[x, y] = 1
            else:
                del detected_object

        # print("Detected: {}, filtered: {}".format(len(obj_grid_list), len(detected_object_list)))
        return detected_object_list, filtered_actor_grid

    @staticmethod
    def isolated_object_detection_fast(lidar_data, depth_map_min, z_threshold, ego_Trans, grid_to_search=None):
        filtered_actor_grid = np.zeros(shape=(LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE), dtype=int)
        if grid_to_search is None:
            grids_to_search = np.array(depth_map_min <= z_threshold)
        else:
            grids_to_search = np.array(grid_to_search)  # cuz this function mutate the drivable grid as check marks

        # number of isolated islands
        ans = 0
        detected_object_list = []
        mvs = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        # print(grid_to_search.shape)
        for i in range(grids_to_search.shape[0]):
            for j in range(grids_to_search.shape[1]):
                # print(grids_to_search[i, j])
                if not grids_to_search[i, j]:
                    continue
                # TODO: uncomment if carla lane type semantic becomes better
                # drivable,_ = LidarPreprocessor.is_drivable_grid(i, j, ego_Trans)
                # if not drivable:
                #     continue
                stack = [[i, j]]
                detected_object = DetectedObject(ans)
                while (stack):
                    pos = stack.pop()
                    grids_to_search[pos[0], pos[1]] = 0
                    # if (pos[0] not in occupancy_grid) or (pos[1] not in occupancy_grid[pos[0]]):
                    #     continue
                    detected_object.insert_occupancy_grid(pos[0], pos[1], depth_map_min[pos[0]][pos[1]])

                    for mv in mvs:
                        x, y = pos[0] + mv[0], pos[1] + mv[1]
                        if 0 <= x < grids_to_search.shape[0] and 0 <= y < grids_to_search.shape[1] and grids_to_search[
                            x, y]:
                            stack.append([x, y])
                """ filter by area and point clouds"""
                if len(detected_object.occupancy_grid_list) > 2 or len(detected_object.point_cloud_list) > 20:
                    ans += 1
                    detected_object.get_bounding_box(z_threshold)
                    detected_object_list.append(detected_object)
                    for [x, y, z] in detected_object.occupancy_grid_list:
                        filtered_actor_grid[x, y] = 1

                    filtered_object_points = LidarPreprocessor.filter_object_points_by_bbox(lidar_data,
                                                                                            detected_object.x_min,
                                                                                            detected_object.x_max,
                                                                                            detected_object.y_min,
                                                                                            detected_object.y_max)
                    if filtered_object_points is not None:
                        detected_object.insert_point_cloud(filtered_object_points.tolist())
        return detected_object_list, filtered_actor_grid

    @staticmethod
    def estimated_actor_and_speed_from_detected_object(detected_object_list, ego_actor, actor_list=None,
                                                       DistanceThresh=3):
        """
        TODO: actor list should be given by an 3D point cloud object detector
        by default: actor list is vehicle only
        """
        if actor_list is None:
            actor_list = CarlaDataProvider.get_world().get_actors()
            actor_list = actor_list.filter("*vehicle*")
        filtered_object_list = []
        for ObjectId in range(len(detected_object_list)):
            min_distance = 10
            min_other_actor_speed = []
            min_other_actor_accel = []
            min_other_actor_position = []
            min_other_actor_id = 0
            for other_actor in actor_list:
                if other_actor.id == ego_actor.id:
                    continue
                another_speed = other_actor.get_velocity()
                other_actor_speed = [another_speed.x, another_speed.y, another_speed.z]
                another_accel = other_actor.get_acceleration()
                other_actor_accel = [another_accel.x, another_accel.y, another_accel.z]

                another_position = other_actor.get_transform().location
                # other_actor.get_transform().distance(hero_actor.get_transform())
                other_actor_position = [another_position.x, another_position.y, another_position.z]
                other_actor_position_numpy_array = np.array([other_actor_position])
                other_actor_position_ego_perspective = \
                    Utils.world_to_car_transform(other_actor_position_numpy_array, ego_actor.get_transform())[0]
                # =print(object_location_list[ObjectId], gd_position_cc)
                detected = False

                # TODO: can use carla.location.distance(carla.location) instead
                detected_object_bbox_center = np.array(detected_object_list[ObjectId].bounding_box[0])
                distance = Utils.get_distance(detected_object_bbox_center[0], other_actor_position_ego_perspective[0],
                                              detected_object_bbox_center[1], other_actor_position_ego_perspective[1])
                if distance <= DistanceThresh and distance < min_distance:
                    min_distance = distance
                    min_other_actor_speed = other_actor_speed
                    min_other_actor_accel = other_actor_accel
                    min_other_actor_position = other_actor_position
                    min_other_actor_id = other_actor.id
            if min_distance < DistanceThresh:
                detected_object_list[ObjectId].set_speed(min_other_actor_speed)
                detected_object_list[ObjectId].set_accel(min_other_actor_accel)
                detected_object_list[ObjectId].set_position(min_other_actor_position)
                detected_object_list[ObjectId].set_actor_id(min_other_actor_id)
                detected_object_list[ObjectId].set_ego_id(ego_actor.id)

                filtered_object_list.append(detected_object_list[ObjectId])
            # else:
            #     detected_object_list[ObjectId].set_speed(min_other_actor_speed)
            #     detected_object_list[ObjectId].set_accel(min_other_actor_accel)
            #     detected_object_list[ObjectId].set_position(min_other_actor_position)
            #     detected_object_list[ObjectId].set_actor_id(min_other_actor_id)

        # # debug
        # for obj in filtered_object_list:
        #     obj.print()
        if ego_actor.id == CarlaActorPool.get_hero_actor().id:
            Utils.summarize_detected_object(filtered_object_list)

        return filtered_object_list

    # # drivable space check
    # @staticmethod
    # def drivable_lanetype_grids():
    #     grid_drivable_lanetype = np.zeros(shape=(LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE), dtype=int)
    #     ego_Trans = CarlaActorPool.get_hero_actor().get_transform()
    #     for i in range(LidarPreprocessor.X_SIZE):
    #         for j in range(LidarPreprocessor.Y_SIZE):
    #             if LidarPreprocessor.is_drivable_grid(i, j, ego_Trans):
    #                 grid_drivable_lanetype[i, j] = 1
    #     return grid_drivable_lanetype

    # drivable space check
    @staticmethod
    def is_valid_grid_lanetype(i, j, ego_Trans, valid_lane_type=Utils.drivable_lane_type, debug=False):

        drivable = False
        node_dist = [LidarPreprocessor.getX_meter(i) + LidarPreprocessor.dX / 2,
                     LidarPreprocessor.getY_meter(j) + LidarPreprocessor.dY / 2, 0]
        # compensate for grid center
        node_loc = Utils.car_to_world_transform(np.array([node_dist]), ego_Trans)
        node_loc_carla = carla.Location(x=node_loc[0, 0], y=node_loc[0, 1], z=node_loc[0, 2])
        # node_waypoint = CarlaDataProvider.get_map().get_waypoint(node_loc_carla, project_to_road=None, lane_type=carla.LaneType.Any)
        node_waypoint = CarlaDataProvider.get_map().get_waypoint(node_loc_carla, project_to_road=None,
                                                                 lane_type=valid_lane_type)

        if node_waypoint is None:
            if debug:
                node_waypoint = CarlaDataProvider.get_map().get_waypoint(node_loc_carla, project_to_road=None,
                                                                         lane_type=carla.LaneType.Any)
                if node_waypoint is not None:
                    print("None valid grid ({},{}) lane type {} at (query {}), ({}) + ({})".format(i, j,
                                                                                                   node_waypoint.lane_type,
                                                                                                   node_loc_carla,
                                                                                                   ego_Trans.location,
                                                                                                   node_dist))
                else:
                    print("None valid grid ({},{}) lane type None at (query {}), ({}) + ({})".format(i, j,
                                                                                                     node_loc_carla,
                                                                                                     ego_Trans.location,
                                                                                                     node_dist))
            return False, None

        # if node_waypoint.lane_type in Utils.valid_lane_type or node_waypoint.lane_type in valid_lane_type:
        #     drivable = True
        #     if debug: print("Driving grid ({},{}) lane type {} at {}(query {}), ({}) + ({})".format(i, j,
        #                                                                                        node_waypoint.lane_type,
        #                                                                                        node_waypoint.transform.location,
        #                                                                                        node_loc_carla,
        #                                                                                        ego_Trans.location,
        #                                                                                        node_dist))
        else:
            if node_waypoint.lane_type != carla.LaneType.Driving:
                if debug: print(
                    "Valid grid ({},{}) lane type {} at {}(query {}), ({}) + ({})".format(i, j, node_waypoint.lane_type,
                                                                                          node_waypoint.transform.location,
                                                                                          node_loc_carla,
                                                                                          ego_Trans.location,
                                                                                          node_dist))
            return True, node_waypoint.lane_type

    @staticmethod
    def save_binary_occupancy_grid(outputdir, occupancy_grid):
        if occupancy_grid is None:
            return
        res = np.array(occupancy_grid.transpose(), dtype=np.uint8)
        res = res * 255
        w = res.shape[0]
        h = res.shape[1]
        res[int(w / 2), int(h / 2)] = 255
        imageio.imwrite(outputdir, res)

    # @staticmethod
    # def point_process(lidar_data, length, width):
    #     try:
    #         x, y, z = lidar_data
    #     except Exception as e:
    #         raise
    #
    #     if LidarPreprocessor.is_self_lidar_data(x, y, length, width):
    #         return
    #
    #     x = LidarPreprocessor.getX(x)
    #     y = LidarPreprocessor.getY(y)
    #
    #     occupancy_grid = defaultdict(dict)
    #     depth_map_min = defaultdict(dict)
    #     depth_map_max = np.zeros((LidarPreprocessor.X_SIZE, LidarPreprocessor.Y_SIZE, 1)) - 10
    #
    #     if x in occupancy_grid.keys() and y in occupancy_grid[x].keys():
    #         occupancy_grid[x][y] += [lidar_data]
    #         depth_map_min[x][y] = min(z, depth_map_min[x][y])
    #     else:
    #         occupancy_grid[x][y] = [lidar_data]
    #         depth_map_min[x][y] = z
    #
    #     depth_map_max[x][y][0] = max(z, depth_map_max[x][y][0])
    #
    #
    # @staticmethod
    # def point_process_parallel(lidar_data_batch, length, width):
    #     occupancy_grid = defaultdict(dict)
    #     for lidar_data in lidar_data_batch:
    #         try:
    #             x, y, z = lidar_data
    #         except Exception as e:
    #             raise
    #
    #         if LidarPreprocessor.is_self_lidar_data(x, y, length, width):
    #             continue
    #
    #         x = LidarPreprocessor.getX(x)
    #         y = LidarPreprocessor.getY(y)
    #
    #         if x in occupancy_grid.keys() and y in occupancy_grid[x].keys():
    #             occupancy_grid[x][y] += [lidar_data]
    #         else:
    #             occupancy_grid[x][y] = [lidar_data]
    #     return occupancy_grid
