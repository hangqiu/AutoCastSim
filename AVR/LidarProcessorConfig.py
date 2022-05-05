
class LidarProcessorConfig():
    X_max = 70
    X_min = -70
    Y_max = 70
    Y_min = -70
    Z_max = 2.5
    Z_min = -2.5
    dX = 0.5
    dY = 0.5
    dZ = 0.5

    X_SIZE = int((X_max - X_min) / dX)
    Y_SIZE = int((Y_max - Y_min) / dY)
    Z_SIZE = int((Z_max - Z_min) / dZ)

    lidar_dim = [X_SIZE, Y_SIZE, Z_SIZE]
    lidar_depth_dim = [X_SIZE, Y_SIZE, 1]