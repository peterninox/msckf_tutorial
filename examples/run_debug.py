import numpy as np
import scipy
from msckf import MSCKF
from msckf_types import CameraCalibration, IMUData, PinholeIntrinsics
from params import AlgorithmConfig, EurocDatasetCalibrationParams
import sys

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

camera_calib = CameraCalibration()
camera_calib.set_extrinsics(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

config = AlgorithmConfig()
config.msckf_params.max_track_length = 3
transition_method = AlgorithmConfig.MSCKFParams.StateTransitionIntegrationMethod
config.msckf_params.state_transition_integration = transition_method.Euler

config.msckf_params.transition_matrix_method = AlgorithmConfig.MSCKFParams.TransitionMatrixModel.discrete

msckf = MSCKF(config.msckf_params, camera_calib)

print(f"gravity: {msckf.gravity}")
print(f"imu_t_camera: {camera_calib.imu_t_camera}")
# print(f"initial P\n{msckf.state.covariance}")

# global_R_imu
gt_rot_matrx = np.identity(3)
# global_t_imu
gt_pos = np.array([0.0, 0.0, 0.0])
gt_vel = np.array([1.0, 0.0, 0.0])
gt_bias_gyro = np.array([0.0, 0.0, 0.0])
gt_bias_acc = np.array([0.0, 0.0, 0.0])

msckf.initialize(gt_rot_matrx, gt_pos, gt_vel, gt_bias_acc, gt_bias_gyro)

timestamp_seconds = 0.0
dt_seconds = 0.1

# Intrinsic parameters
K = np.array([[300.0, 0.0, 400.0], [250.0, 0.0, 300.0], [0.0, 0.0, 1.0]])

# Point in global coordinates in front of the camera
pointG = np.array([0.1, -0.2, 2.3])

print(f"Init P\n{msckf.state.covariance}")

for frameID in range(1):
    imu_buffer = []
    for idxImu in range(1):
        acc = np.array([0.05, 0.10, 10.1])
        # gyro = np.array([-0.04, 0.08, -0.03])
        gyro = np.array([0.0, 0.0, 0.0])
        imu_buffer.append(IMUData(acc, gyro, 0, dt_seconds))
        timestamp_seconds += dt_seconds
    msckf.propogate(imu_buffer)

    print(f"P\n{msckf.state.covariance}")

    # print("------------- Propagate")
    # print(f"quat_imu_to_global: {msckf.state.imu_JPLQ_global}")
    # print(f"global_t_imu:       {msckf.state.global_t_imu}")
    # print(f"vel:                {msckf.state.velocity}")
    #
    # ids = []
    # measurements = []
    #
    # # Location of the point in camera frame
    # pointC = pointG - msckf.state.global_t_imu
    #
    # # Append the point to observation list
    # ids.append(0)
    # measurements.append([pointC[0] / pointC[2], pointC[1] / pointC[2]])
    # msckf.add_camera_features(np.array(ids), np.array(measurements))
    #
    # print("------------- Update")
    # print(f"quat_imu_to_global: {msckf.state.imu_JPLQ_global}")
    # print(f"global_t_imu:       {msckf.state.global_t_imu}")
    # print(f"vel:                {msckf.state.velocity}")

print("\nDone!")
