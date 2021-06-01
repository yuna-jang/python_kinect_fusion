import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

import fusion

# Kinect module
import pyk4a
from helpers import convert_to_bgra_if_required
from pyk4a import Config, PyK4A
from pyk4a import PyK4APlayback

from icp_modules.ICP import *
from icp_modules.ICP_algorithm import *
# from icp_modules.ICP_kms import *
from icp_modules.FramePreprocessing import PointCloud

from helpers import colorize, convert_to_bgra_if_required

if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #

  ## Open kinect camera by realtime
  k4a = PyK4A(
    Config(
      color_resolution=pyk4a.ColorResolution.RES_720P,
      depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
      color_format=pyk4a.ImageFormat.COLOR_MJPG
    )
  )

  # Load video file
  filename = r'C:\Users\82106\PycharmProjects\dino_lib\python_kinect_fusion\video1.mkv'
  filename = r'0_sample_video\0531\0531_3.mkv'
  n_frames = 100

  k4a = PyK4APlayback(filename)
  k4a.open()

  # Load Kinect's intrinsic parameter
  cam_intr = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)

  # List 생성
  list_depth_im = []
  list_color_im = []
  # vol_bnds 생성
  vol_bnds = np.zeros((3, 2))
  voxel_size = 0.02
  iter = 0
  # while True:
  for i in range(0,n_frames):
      capture = k4a.get_next_capture()
      if capture.depth is not None and capture.color is not None: #and i%3==0:
        print(f"==========={iter}==========")

        # Read depth and color image
        depth_im = capture.transformed_depth.astype(float)
        depth_im /= 1000.  ## depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im >= 3] = 0  # set invalid depth to 0 (specific to 7-scenes dataset) 65.535=2^16/1000
        color_capture = convert_to_bgra_if_required(k4a.configuration["color_format"], capture.color)
        color_im = cv2.cvtColor(color_capture, cv2.COLOR_BGR2RGB)

        filtered_depth = cv2.bilateralFilter(depth_im.astype(np.float32), 5, 35, 35)
        filtered_color = cv2.bilateralFilter(color_im, 10, 50, 50)
        sharpening_2 = np.array([[-1, -1, -1, -1, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, 2, 9, 2, -1],
                                 [-1, 2, 2, 2, -1],
                                 [-1, -1, -1, -1, -1]]) / 9.0
        filtered_color = cv2.filter2D(filtered_color, -1, sharpening_2)

        # cv2.imwrite(fr'0_sample_video\frames\{i}_depth.jpg', filtered_depth)
        cv2.imwrite(fr'0_sample_video\frames\{i}_color.jpg', filtered_color)

        list_depth_im.append(depth_im)
        list_color_im.append(filtered_color)

        if iter == 0:
          first_Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))
          cam_pose = np.eye(4)
          first_pose = cam_pose

        elif iter >= 1:
          second_Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))
          ind = random.sample(range(first_Points3D.shape[1]), second_Points3D.shape[1])
          pose, distances, _ = icp(second_Points3D.T, first_Points3D.T[ind, :])  # A, B // maps A onto B : B = pose*A
          cam_pose = np.dot(first_pose, pose)

          first_pose = cam_pose
          first_Points3D = second_Points3D

          # Compute camera view frustum and extend convex hull
          view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
          vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
          vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

        iter = iter + 1

  # ======================================================================================================== #
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)
  k4a.close()

  # ===============Integrate===============
  n_imgs = len(list_depth_im)
  iter=0
  for iter in range(0, n_imgs):
    print("Fusing frame %d/%d"%(iter+1, n_imgs))

    # Read depth and color image
    depth_im = list_depth_im[iter]
    color_im = list_color_im[iter]

    # Set first frame as world system
    if iter == 0:
      previous_Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))
      cam_pose = np.eye(4)
      previous_pose = cam_pose

    elif iter >= 1:
      Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))

      # Compute camera view frustum and extend convex hull
      pose, distances, _ = icp(Points3D.T, previous_Points3D.T)  # A, B // maps A onto B : B = pose*A
      pose = np.dot(previous_pose, pose)

      view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, pose)
      vol_bnds_seq = np.zeros((3, 2))
      vol_bnds_seq[:, 0] = np.minimum(vol_bnds_seq[:, 0], np.amin(view_frust_pts, axis=1))
      vol_bnds_seq[:, 1] = np.maximum(vol_bnds_seq[:, 1], np.amax(view_frust_pts, axis=1))
      tsdf_vol_seq = fusion.TSDFVolume(vol_bnds_seq, voxel_size=voxel_size)
      tsdf_vol_seq.integrate(color_im, depth_im, cam_intr, pose, obs_weight=1.)
      second_Points3D = tsdf_vol_seq.get_point_cloud()[:, 0:3]

      # 누적 pointcloud vertex only
      # first_Points3D = tsdf_vol.get_partial_point_cloud()
      first_Points3D = tsdf_vol.get_point_cloud()[:, 0:3]

      pts_size = min(first_Points3D.shape[0], second_Points3D.shape[0])
      size = min(pts_size, 30000)
      samples_first = random.sample(range(first_Points3D.shape[0]), size)
      samples_second = random.sample(range(second_Points3D.shape[0]), size)
      pose_real, _, _ = icp(second_Points3D[samples_second, :],first_Points3D[samples_first, :])  # A, B // maps A onto B : B = pose*A
      pose_real = np.dot(previous_pose, pose_real)

      # # icp points 검증
      # fig = plt.figure(figsize=(8, 8))
      # ax = fig.add_subplot(projection='3d')  # Axe3D object
      # ax.scatter(second_Points3D[samples_second, 0], second_Points3D[samples_second, 1], second_Points3D[samples_second, 2], color='r', s=0.3)
      # ax.scatter(first_Points3D[samples_first, 0], first_Points3D[samples_first, 1], first_Points3D[samples_first, 2], color='b', s=0.3)
      # plt.show()
      #
      # # pose matrix 검증
      # fig = plt.figure(figsize=(8, 8))
      # ax = fig.add_subplot(projection='3d')  # Axe3D object
      # P = np.vstack((second_Points3D.T, np.ones((1, second_Points3D.T.shape[1]))))  # projection
      # proj = np.dot(pose_real, P)
      # # ax.scatter(P.T[samples, 0], P.T[samples, 1], P.T[samples, 2], color='r', s=0.3)
      # # ax.scatter(proj.T[samples, 0], proj.T[samples, 1], proj.T[samples, 2], color='g', s=0.3)
      # # ax.scatter(first_Points3D[samples, 0], first_Points3D[samples, 1], first_Points3D[samples, 2], color='b', s=0.3)
      # ax.scatter(proj.T[:, 0], proj.T[:, 1], proj.T[:, 2], color='g', s=0.3)
      # ax.scatter(first_Points3D[:, 0], first_Points3D[:, 1], first_Points3D[:, 2], color='b', s=0.3)
      # plt.show()

      cam_pose = pose_real
      previous_pose = cam_pose
      previous_Points3D = Points3D

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_im, depth_im, cam_intr, cam_pose, obs_weight=1.)
    iter=iter+1


  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to test_mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("test_mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to test_pcd.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("test_pcd.ply", point_cloud)