# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
import carla
import re
import pygame
import math
import numpy as np
from AVR.PCProcess import LidarPreprocessor
from AVR.autocast_agents.AStarPlanner import AStarPlanner
from AVR import Utils
# from srunner.scenariomanager.carla_data_provider import CarlaActorPool
from srunner.scenariomanager.carla_data_provider import CarlaActorPool


def simple_algorithm(occupancy_grid, source, destination):
    result = []
    interval = 5.0 # this should be modified as it should be controlled by the speed
    x_diff = destination[0] - source[0]
    y_diff = destination[1] - source[1]
    for i in range(int(interval)):
        result += [(source[0]+x_diff/interval*(i+1), source[1]+y_diff/interval*(i+1), 0)]
    print(result)
    return result


def plan_trajectory_with_timestamp_astar(lidar_proc_res, grid_thresh, route_to_destination, interpolated_loc, frameId, myTrans, mySpeed):
    a_star = AStarPlanner()
    # myTrans = CarlaActorPool.get_hero_actor().get_transform()

    route_to_destination_ego_perspective = []
    for waypoint_loc in route_to_destination:
        destination_m = np.matrix(waypoint_loc)
        destination_m = Utils.world_to_car_transform(destination_m, myTrans)[0]
        route_to_destination_ego_perspective.append(destination_m)

    interpolated_route_to_destination_ego_perspective = []
    for waypoint_trans, roadopt in interpolated_loc:
        waypoint_loc = waypoint_trans.location
        destination_m = np.matrix((waypoint_loc.x, waypoint_loc.y, waypoint_loc.z))
        destination_m = Utils.world_to_car_transform(destination_m, myTrans)[0]
        interpolated_route_to_destination_ego_perspective.append(destination_m)

    """ the interpolate function doesn't have ending node"""
    interpolated_route_to_destination_ego_perspective.append(route_to_destination_ego_perspective[-1])
    # print(route_to_destination_ego_perspective)
    if len(route_to_destination_ego_perspective) == 0:
        print("route empty")
        return [None, None, None, False, False]

    # print("astar.planning")
    rx_result, ry_result, dist_m_result, deviation_dist_m_result, complete_path = a_star.planning(lidar_proc_res, route_to_destination_ego_perspective,
                                                          interpolated_route_to_destination_ego_perspective,
                                                          grid_thresh)

    if rx_result.shape[0] <= len(route_to_destination_ego_perspective):
        return [None, None, None, False, False]
    if (not complete_path) and rx_result.shape[0] < 10:
        return [None, None, None, False, complete_path]
    rz = np.ones(shape=(rx_result.size)) * myTrans.location.z
    result_t = np.matrix(list(zip(rx_result, ry_result, rz)))
    result = Utils.car_to_world_transform(result_t, myTrans)
    result = np.matrix(result.transpose())
    result = np.append(result, np.ones((1, result.shape[1])), axis=0)
    result = np.array(result.transpose())
    result[..., 2] = rz
    """ calculate timestamp based on distance and speed"""
    ego_speed = math.hypot(mySpeed.x, mySpeed.y)
    framestamp = estimate_trajectory_framestamp(frameId, ego_speed, dist_m_result)
    result[..., 3] = framestamp
    # print(dist_m_result)
    # print(ego_speed)
    # print(framestamp)



    return [result.tolist(),  dist_m_result.tolist(), deviation_dist_m_result.tolist(), True, complete_path]

def estimate_trajectory_framestamp(cur_frameId, ego_speed, dist_m_result, target_speed_mps=Utils.target_speed_mps): # 5.56 is 20 km/h
    """ assume linear accelartion
        Dist = (Vt-Vo)^2 / 2a
        t= (sqrt(2aD + Vo^2) - Vo) / a
        asssume 100kmph acc time 4 sec,  a = 7m/s/s, normaly just 2
    """
    a = 2
    # print(dist_m_result.shape)
    timestamp = (np.sqrt(2*a*dist_m_result+ego_speed**2) - ego_speed) / a
    framestamp = timestamp / 0.1 + cur_frameId
    """ max speed is target speed"""
    max_dist = (target_speed_mps ** 2 - ego_speed ** 2) / 2 / a
    last_dist_before_max_dist = 0
    last_dist_index = 0
    for i, d in enumerate(dist_m_result):
        if d < max_dist:
            last_dist_before_max_dist = d
            last_dist_index = i
        else:
            framestamp[i] = framestamp[last_dist_index] + (d - last_dist_before_max_dist) / target_speed_mps / 0.1

    return framestamp


def calculate_trajectory_crossing(ego_trajectory_points_timestamp, ego_deviation_for_collision, detected_obj,
                                  ego_spd_val_mps, collision_points_threshold=1):

    """
    planaheadframe = 20  # min 2 second rule (good for 20km/h), each frame is 100 ms, max propotional to speed
    @param ego_trajectory_points_timestamp:
    @param ego_deviation_for_collision:
    @param detected_obj:
    @param collision_points_threshold:
    @return: collider trajectory, collision detected, clear to proceed
    """
    debug = False
    ref_spd = 20 / 3.6
    planaheadframe = 40
    planaheadframe_fortargetspeed = planaheadframe / ref_spd * ego_spd_val_mps
    planaheadframe = max(planaheadframe, planaheadframe_fortargetspeed)

    collision_detected = False
    trajectory_clear = True
    collision_detail = None
    collider_speed = detected_obj.estimated_speed
    collider_spd_val_mps = math.hypot(collider_speed[0], collider_speed[1])
    another_accel = detected_obj.estimated_accel
    another_position = detected_obj.esitmated_position
    if collider_speed is None or (abs(collider_speed[0]) < 0.01 and abs(collider_speed[1]) < 0.01):
        # print("speed {} too low".format(another_speed))
        return [[], collision_detected, trajectory_clear, collision_detail]

    # timeframe size is 0.1 s = 100 ms

    # frametime_bound = 10 # upper + lower  = 2 seconds
    vehicle_length = 6
    max_passthru_frametime = 2 * Utils.CarlaFPS # max 2 seconds
    ego_passthru_frametime = max_passthru_frametime
    collider_passthru_frametime = max_passthru_frametime
    if ego_spd_val_mps != 0:
        ego_passthru_frametime = min(max_passthru_frametime, int(vehicle_length / ego_spd_val_mps * Utils.CarlaFPS))
    if collider_spd_val_mps !=0:
        collider_passthru_frametime = min(max_passthru_frametime, int(vehicle_length / collider_spd_val_mps * Utils.CarlaFPS))

    loc_bound = 2 # was 0.5 meter, 1 grid
    deviation_threshold = LidarPreprocessor.dY * 10 # meters of X grids
    waypoints_size = max(len(ego_trajectory_points_timestamp), 50) # at least 50 points to predict 5 sec
    ego_trajectory_points_timestamp = np.array(ego_trajectory_points_timestamp)
    # print(waypoints.shape)
    new_list = np.arange(0, waypoints_size, 1)
    time_step = 0.1 # 100ms
    collider_trajectory_x = another_position[0] + collider_speed[0]*time_step*new_list
    collider_trajectory_y = another_position[1] + collider_speed[1]*time_step*new_list
    # with acceleration
    # collider_trajectory_x = another_position[0] + another_speed[0] * time_step * new_list + 1/2 * another_accel[0] * ((time_step * new_list) ** 2)
    # collider_trajectory_y = another_position[1] + another_speed[1] * time_step * new_list + 1/2 * another_accel[1] * ((time_step * new_list) ** 2)
    collider_trajectory_z = np.zeros(shape=collider_trajectory_y.shape)
    start_frame = ego_trajectory_points_timestamp[0][3]
    collider_trajectory_frameid = start_frame + new_list
    collider_trajectory = np.array(list(zip(collider_trajectory_x, collider_trajectory_y, collider_trajectory_z, collider_trajectory_frameid)))
    cur_location_x = ego_trajectory_points_timestamp[0][0]
    cur_location_y = ego_trajectory_points_timestamp[0][1]
    # print(predict_trajectory)
    deviate_start_frame = -1
    deviate_start_xy = None

    for item_t in range(len(ego_trajectory_points_timestamp)):
        # consider only upto planaheadframe into future, o.w. don't stop even if colliding
        # exception: if planned route deviates from default, than consider it till the point when it merge back to default, if colliding and the deviation start is within planahead frame, then stop otherwise not stopping
        wp_x = ego_trajectory_points_timestamp[item_t][0]
        wp_y = ego_trajectory_points_timestamp[item_t][1]
        wp_frame = ego_trajectory_points_timestamp[item_t][3]

        if len(ego_deviation_for_collision) > item_t:
            if ego_deviation_for_collision[item_t] >= deviation_threshold and deviate_start_frame == -1:
                deviate_start_frame = wp_frame - start_frame
                deviate_start_xy = [wp_x, wp_y]
                if debug: print("Deviation starts at waypoint {} (frame {})".format(item_t, wp_frame-start_frame))
            if ego_deviation_for_collision[item_t] < deviation_threshold and deviate_start_frame != -1:
                deviate_start_frame = -1
                deviate_start_xy = None
                if debug: print("Deviation ends at waypoint {} (frame {})".format(item_t, wp_frame-start_frame))


        upper_x = wp_x + loc_bound
        lower_x = wp_x - loc_bound
        upper_y = wp_y + loc_bound
        lower_y = wp_y - loc_bound
        # upper_t = cur_frame + frametime_bound
        # lower_t = cur_frame - frametime_bound
        upper_t = wp_frame + ego_passthru_frametime
        lower_t = wp_frame

        # print("Ego time range: [{}, {}]".format(lower_t, upper_t))
        ego_wp_loc = [wp_x, wp_y]
        ego_time_window = [lower_t, upper_t]
        collider_wp_loc = []
        collider_time_window = []

        colliding_waypoints_count = 0
        for collider_t in range(len(collider_trajectory_x)):
            if lower_x < collider_trajectory_x[collider_t] < upper_x \
                    and lower_y < collider_trajectory_y[collider_t] < upper_y :
                """ Make an exception for trajectories that cross ego center (idx=0), cuz they will stop """
                if item_t < 3:
                    if debug: print("Collider coming to ego center")
                    return [list(collider_trajectory), collision_detected, trajectory_clear, collision_detail]
                if debug: print("Spatial collision point index {}".format(collider_t))
                if debug: print("Ego time range: [{}, {}]".format(lower_t, upper_t))
                if debug: print("Collider time range: [{}, {}]".format(collider_trajectory_frameid[collider_t],
                                                                       collider_trajectory_frameid[
                                                                           collider_t] + collider_passthru_frametime))
                # if lower_t <= collider_trajectory_frameid[collider_t] <= upper_t \
                #         or lower_t <= collider_trajectory_frameid[collider_t] + collider_passthru_frametime <= upper_t:
                if collider_trajectory_frameid[collider_t] > upper_t or collider_trajectory_frameid[collider_t] + collider_passthru_frametime < lower_t:
                    continue
                else:
                    colliding_waypoints_count += 1
                    if colliding_waypoints_count >= collision_points_threshold:
                        collider_wp_loc = [collider_trajectory_x[collider_t], collider_trajectory_y[collider_t]]
                        collider_time_window = [collider_trajectory_frameid[collider_t], collider_trajectory_frameid[collider_t]+collider_passthru_frametime]
                        break

        if colliding_waypoints_count >= collision_points_threshold:
            collision_location_x = wp_x
            collision_location_y = wp_y
            collision_detected = True
            if wp_frame - start_frame > planaheadframe:
                if deviate_start_frame == -1:
                    if debug: print("Stop considering waypoint {} (frame {}), no deviation".format(item_t, wp_frame-start_frame))
                    break
                if deviate_start_frame > planaheadframe:
                    if debug: print("Stop considering waypoint {} (frame {}), deviation starts too late".format(item_t, wp_frame-start_frame))
                    break
            trajectory_clear = False
            if debug: print("Collider Actor ID: {}".format(detected_obj.actor_id))
            # to collision point
            ego_dist_to_collision = Utils.get_distance(collision_location_x, cur_location_x, collision_location_y, cur_location_y)
            collider_dist_to_collision = Utils.get_distance(collision_location_x, another_position[0], collision_location_y, another_position[1])
            if debug:
                print("Ego distance to collision: " + str(ego_dist_to_collision))
                print("Collider distance to collision: " + str(collider_dist_to_collision))
                print("Time to collision: {} sec, Planned Frame Ahead: {}".format((wp_frame - start_frame) * 0.1,
                                                                              wp_frame - start_frame))
            # to deviation point
            ego_dist_to_deviation = -1
            collider_dist_to_deviation = -1
            if deviate_start_frame != -1:
                ego_dist_to_deviation = Utils.get_distance(deviate_start_xy[0], cur_location_x, deviate_start_xy[1], cur_location_y)
                collider_dist_to_deviation = Utils.get_distance(deviate_start_xy[0], another_position[0], deviate_start_xy[1],
                                       another_position[1])
                if debug:
                    print("Ego distance to deviation: " + str(ego_dist_to_deviation))
                    print("Collider distance to deviation: " + str(collider_dist_to_deviation))
                    print("Time to deviation: {} sec, Planned Frame Ahead: {}".format((wp_frame - deviate_start_frame) * 0.1,
                                                                                  wp_frame - deviate_start_frame))

            collision_detail = CollisionDetail(CarlaActorPool.get_hero_actor().id, detected_obj.actor_id,
                                               ego_dist_to_collision, collider_dist_to_collision, ego_dist_to_deviation,
                                               collider_dist_to_deviation, wp_frame, start_frame, deviate_start_frame,
                                               ego_wp_loc, collider_wp_loc, ego_time_window, collider_time_window)

            break
    return [list(collider_trajectory), collision_detected, trajectory_clear, collision_detail]

class CollisionDetail(object):
    def __init__(self, ego_id, collider_id, ego_distance_to_collision, collider_distance_to_collision, ego_distance_to_deviation,
                        collider_distance_to_deviation, collision_frame, start_frame, deviate_start_frame,
                        ego_wp_loc, collider_wp_loc, ego_time_window, collider_time_window):
        self.ego_id = ego_id
        self.collider_id = collider_id
        self.ego_distance_to_collision = ego_distance_to_collision
        self.collider_distance_to_collision = collider_distance_to_collision
        self.ego_distance_to_deviation = ego_distance_to_deviation
        self.collider_distance_to_deviation = collider_distance_to_deviation
        self.ego_frames_to_collision = collision_frame - start_frame
        self.ego_frames_to_deviation = deviate_start_frame - start_frame
        self.ego_sec_to_collision = self.ego_frames_to_collision * 0.1
        self.ego_sec_to_deviation = self.ego_frames_to_deviation * 0.1
        self.ego_wp_loc = ego_wp_loc
        self.collider_wp_loc = collider_wp_loc
        self.ego_time_window = ego_time_window
        self.collider_time_window = collider_time_window

    # def __str__(self):
    #     return "Ego: {}({}m, {}m)\nCollider:{} ({}m, {}m)\nETA: {}sec\n".format(self.ego_id,
    #                                                                             self.ego_distance_to_collision,
    #                                                                             self.ego_distance_to_deviation,
    #                                                                             self.collider_id,
    #                                                                             self.collider_distance_to_collision,
    #                                                                             self.collider_distance_to_deviation,
    #                                                                             self.ego_sec_to_collision)

    def to_str_list(self):

        return [" Ego {} wp:{}({}m,{}m)".format(self.ego_id, self.ego_wp_loc, round(self.ego_distance_to_collision, 2), round(self.ego_distance_to_deviation, 2)),
                " Collider {} wp:{}({}m,{}m)".format(self.collider_id, self.collider_wp_loc, round(self.collider_distance_to_collision, 2), round(self.collider_distance_to_deviation, 2)),
                " TW {}/{}".format(self.ego_time_window, self.collider_time_window),
                " ETA: {} sec".format(round(self.ego_sec_to_collision, 2))]

# def calculate_trajectory_crossing(ego_trajectory_points_timestamp, ego_deviation_for_collision, detected_obj, collision_points_threshold=1):
#     """
#
#     @param ego_trajectory_points_timestamp:
#     @param ego_deviation_for_collision:
#     @param detected_obj:
#     @param collision_points_threshold:
#     @return: collider trajectory, collision detected, clear to proceed
#     """
#
#
#     collision_detected = False
#     trajectory_clear = True
#     another_speed = detected_obj.estimated_speed
#     another_position = detected_obj.esitmated_position
#     if abs(another_speed[0]) < 0.01 and abs(another_speed[1]) < 0.01:
#         # print("speed {} too low".format(another_speed))
#         return [[], collision_detected, trajectory_clear]
#
#     # timeframe size is 0.1 s = 100 ms
#     planaheadframe = 20  # 2 second rule, each frame is 100 ms
#     loc_bound = 1
#     frametime_bound = 10 # upper + lower  = 2 seconds
#     deviation_threshold = LidarPreprocessor.dY * 3 # meters of 2 grids
#     waypoints_size = max(len(ego_trajectory_points_timestamp), 50) # at least 50 points to predict 5 sec
#     ego_trajectory_points_timestamp = np.array(ego_trajectory_points_timestamp)
#     # print(waypoints.shape)
#     new_list = np.arange(0, waypoints_size, 1)
#     time_step = 0.1 # 100ms
#     collider_trajectory_x = another_position[0] + new_list*another_speed[0]*time_step
#     collider_trajectory_y = another_position[1] + new_list*another_speed[1]*time_step
#     collider_trajectory_z = np.zeros(shape=collider_trajectory_y.shape)
#     collider_trajectory_frameid = ego_trajectory_points_timestamp[0][3] + new_list
#     collider_trajectory = np.array(list(zip(collider_trajectory_x, collider_trajectory_y, collider_trajectory_z, collider_trajectory_frameid)))
#     cur_location_x = ego_trajectory_points_timestamp[0][0]
#     cur_location_y = ego_trajectory_points_timestamp[0][1]
#     cur_frame = ego_trajectory_points_timestamp[0][3]
#     # print(predict_trajectory)
#     deviate_start_frame = -1
#     for item_t in range(len(collider_trajectory_x)):
#         # consider only upto planaheadframe into future, o.w. don't stop even if colliding
#         # exception: if planned route deviates from default, than consider it till the point when it merge back to default, if colliding and the deviation start is within planahead frame, then stop otherwise not stopping
#         if len(ego_deviation_for_collision) > item_t:
#             if ego_deviation_for_collision[item_t] >= deviation_threshold and deviate_start_frame == -1:
#                 deviate_start_frame = item_t
#                 print("Deviation starts at frame {}".format(item_t))
#             if ego_deviation_for_collision[item_t] < deviation_threshold and deviate_start_frame != -1:
#                 deviate_start_frame = -1
#                 print("Deviation ends at frame {}".format(item_t))
#
#         upper_x = collider_trajectory_x[item_t] + loc_bound
#         lower_x = collider_trajectory_x[item_t] - loc_bound
#         upper_y = collider_trajectory_y[item_t] + loc_bound
#         lower_y = collider_trajectory_y[item_t] - loc_bound
#         upper_t = collider_trajectory_frameid[item_t] + frametime_bound
#         lower_t = collider_trajectory_frameid[item_t] - frametime_bound
#         waypoints_temp = ego_trajectory_points_timestamp
#
#         for i in [[-1, 0, 0, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 1, 0, 0]]:
#             waypoints_new = ego_trajectory_points_timestamp + np.array(i) * loc_bound
#             waypoints_temp = np.concatenate((waypoints_temp, waypoints_new), axis=0)
#         filtered_waypoints = waypoints_temp[(waypoints_temp[..., 0] <= upper_x) & (waypoints_temp[..., 0] >= lower_x) &
#                                         (waypoints_temp[..., 1] <= upper_y) & (waypoints_temp[..., 1] >= lower_y) &
#                                         (waypoints_temp[..., 3] <= upper_t) & (waypoints_temp[..., 3] >= lower_t)]
#         if filtered_waypoints.shape[0] >= collision_points_threshold:
#             collision_location_x = collider_trajectory_x[item_t] #np.mean(waypoints_temp[..., 0]) #
#             collision_location_y = collider_trajectory_y[item_t] #np.mean(waypoints_temp[..., 1])#
#             collision_detected = True
#             if item_t > planaheadframe:
#                 if deviate_start_frame == -1:
#                     print("Stop considering frame {}, no deviation".format(item_t))
#                     break
#                 if deviate_start_frame > planaheadframe:
#                     print("Stop considering frame {}, deviation starts too late".format(item_t))
#                     break
#             trajectory_clear = False
#             print("Collider Actor ID: {}".format(detected_obj.actor_id))
#             print("Ego distance to collision: " + str(Utils.get_distance(collision_location_x, cur_location_x, collision_location_y, cur_location_y)))
#             print("Collider distance to collision: " + str(
#                 Utils.get_distance(collision_location_x, another_position[0], collision_location_y, another_position[1])))
#             print("Time to collision: {} sec, Planned Frame Ahead: {}".format(item_t * 0.1, item_t))
#             break
#     return [list(collider_trajectory), collision_detected, trajectory_clear]
#
#
#
#
#
