import numpy as np
from AVR.LidarProcessorConfig import LidarProcessorConfig

class DetectedObject(object):
    def __init__(self, id):
        self.id = id
        self.ego_id = None
        self.occupancy_grid_list = []
        self.point_cloud_list = []
        self.bounding_box = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]  # center, dimension
        self.x_min = 0
        self.x_max = 0
        self.y_min = 0
        self.y_max = 0
        self.z_min = 0
        self.z_max = 0
        self.estimated_speed = None
        self.estimated_accel = None
        self.esitmated_position = None
        self.actor_id = None
        self.obj_for_comm = ObjectForComm()

    def get_obj_for_comm(self):
        self.obj_for_comm.id = self.id
        self.obj_for_comm.occupancy_grid_list = self.occupancy_grid_list
        self.obj_for_comm.point_cloud_list = self.point_cloud_list
        self.obj_for_comm.bounding_box = self.bounding_box
        self.obj_for_comm.estimated_speed = self.estimated_speed
        self.obj_for_comm.estimated_accel = self.estimated_accel
        self.obj_for_comm.esitmated_position = self.esitmated_position
        self.obj_for_comm.actor_id = self.actor_id
        self.obj_for_comm.ego_id = self.ego_id
        return self.obj_for_comm

    def insert_occupancy_grid(self, x, y, z):
        self.occupancy_grid_list.append([x, y, float(z)])

    def insert_point_cloud(self, point_cloud):
        self.point_cloud_list += point_cloud

    def get_bounding_box(self, z_threshold):
        if len(self.occupancy_grid_list)==0:
            return
        occupancy_grid_list_np = np.array(self.occupancy_grid_list)
        self.x_min = round((min(occupancy_grid_list_np[:, 0]) * LidarProcessorConfig.dX + LidarProcessorConfig.X_min), 2)
        self.x_max = round((max(occupancy_grid_list_np[:, 0]) * LidarProcessorConfig.dX + LidarProcessorConfig.X_min), 2)
        self.y_min = round((min(occupancy_grid_list_np[:, 1]) * LidarProcessorConfig.dY + LidarProcessorConfig.Y_min), 2)
        self.y_max = round((max(occupancy_grid_list_np[:, 1]) * LidarProcessorConfig.dY + LidarProcessorConfig.Y_min), 2)
        self.z_min = z_threshold
        self.z_max = max(occupancy_grid_list_np[:, 2])
        self.bounding_box[0][0] = round((self.x_min + self.x_max) / 2.0, 2)
        self.bounding_box[0][1] = round((self.y_min + self.y_max) / 2.0, 2)
        self.bounding_box[0][2] = round((self.z_min + self.z_max) / 2.0, 2)
        self.bounding_box[1][0] = round((self.x_max - self.bounding_box[0][0]),
                                        2) if self.x_min != self.x_max else round(LidarProcessorConfig.dX / 2.0, 2)
        self.bounding_box[1][1] = round((self.y_max - self.bounding_box[0][1]),
                                        2) if self.y_min != self.y_max else round(LidarProcessorConfig.dY / 2.0, 2)
        self.bounding_box[1][2] = round((self.z_max - self.bounding_box[0][2]),
                                        2) if self.z_min != self.z_max else round(LidarProcessorConfig.dZ / 2.0, 2)

    def set_speed(self, speed):
        self.estimated_speed = speed

    def set_accel(self, accel):
        self.estimated_accel = accel

    def set_position(self, pos):
        self.esitmated_position = pos

    def set_actor_id(self, actor_id):
        self.actor_id = actor_id

    def set_ego_id(self, ego_id):
        self.ego_id = ego_id

    def print(self):
        print(
            f"Ego {self.ego_id}, "
            f"Object {self.id} (Actor {self.actor_id}): "
            f"{len(self.occupancy_grid_list)} grids, "
            f"{len(self.point_cloud_list)} points, "
            f"at {self.bounding_box}"
        )


class ObjectForComm(object):
    def __init__(self):
        self.id = None
        self.occupancy_grid_list = []
        self.point_cloud_list = []
        self.bounding_box = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]  # center, dimension
        self.estimated_speed = None
        self.estimated_accel = None
        self.esitmated_position = None
        self.actor_id = None
        self.ego_id=None

    def print(self):
        print(
            f"Ego {self.ego_id}, "
            f"Object {self.id} (Actor {self.actor_id}): "
            f"{len(self.occupancy_grid_list)} grids, "
            f"{len(self.point_cloud_list)} points, "
            f"at {self.bounding_box}"
        )