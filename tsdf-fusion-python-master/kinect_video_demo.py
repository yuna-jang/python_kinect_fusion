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
    )
  )

  # Load video file
  filename = r'0_sample_video\video1.mkv'
  n_frames = 50
  k4a = PyK4APlayback(filename)
  k4a.open()

  # Load Kinect's intrinsic parameter
  cam_intr = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)

  # List 생성
  list_depth_im = []
  list_color_im = []
  # vol_bnds 생성
  vol_bnds = np.zeros((3, 2))
  voxel_size = 0.03
  iter = 0
  # while True:
  for i in range(0,n_frames):
      capture = k4a.get_next_capture()
      if capture.depth is not None and capture.color is not None:
        print(f"==========={iter}==========")
        # Read depth and color image
        depth_im = capture.transformed_depth.astype(float)
        depth_im /= 1000.  ## depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset) 65.535=2^16/1000
        color_im = cv2.cvtColor(capture.color, cv2.COLOR_BGR2RGB)

        list_depth_im.append(depth_im)
        list_color_im.append(color_im)

        if iter == 0:
          first_Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))
          cam_pose = np.eye(4)
          first_pose = cam_pose

        elif iter == 1:
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

    elif iter == 1:
      second_Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))
      pose, distances, _ = icp(second_Points3D.T, previous_Points3D.T)  # A, B // maps A onto B : B = pose*A
      cam_pose = np.dot(previous_pose, pose)
      previous_pose = cam_pose
      previous_Points3D = second_Points3D

    elif iter > 1:
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
      first_Points3D = tsdf_vol.get_partial_point_cloud()

      pts_size = min(first_Points3D.shape[0], second_Points3D.shape[0])
      pose, distances, _ = icp(second_Points3D[0:pts_size, :],
                               first_Points3D[0:pts_size, :])  # A, B // maps A onto B : B = pose*A
      print(f'{pts_size} / {first_Points3D.shape[0]}')
      pose = np.dot(previous_pose, pose)

      cam_pose = pose
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