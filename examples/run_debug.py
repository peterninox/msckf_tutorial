import numpy as np
import scipy
from msckf import MSCKF
from msckf_types import CameraCalibration, IMUData, PinholeIntrinsics
from params import AlgorithmConfig, EurocDatasetCalibrationParams
import sys

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

camera_calib = CameraCalibration()
config = AlgorithmConfig()

msckf = MSCKF(config.msckf_params, camera_calib)

print(f"gravity: {msckf.gravity}")

# global_R_imu
gt_rot_matrx = np.identity(3)
# global_t_imu
gt_pos = np.array([0, 0, 0])
gt_vel = np.array([1, 0, 0])
gt_bias_gyro = np.array([0, 0, 0])
gt_bias_acc = np.array([0, 0, 0])

msckf.initialize(gt_rot_matrx, gt_pos, gt_vel, gt_bias_acc, gt_bias_gyro)

imu_buffer = []

timestamp_seconds = 0.0
dt_seconds = 0.05

for i in range(2):
    acc = np.array([0.05, 0.10, 10.1])
    gyro = np.array([-0.04, 0.08, -0.03])
    imu_buffer.append(IMUData(acc, gyro, timestamp_seconds, dt_seconds))
    timestamp_seconds += dt_seconds

msckf.propogate(imu_buffer)

print(f"global_t_imu={msckf.state.global_t_imu}")
