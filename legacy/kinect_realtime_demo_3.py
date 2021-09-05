"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import matplotlib.pyplot as plt
import cv2
import numpy as np

import fusion

import pyk4a
from helpers import colorize
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
  print("Estimating voxel volume bounds...")
  n_imgs_begin = 800
  n_imgs_end = 850
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  vol_bnds = np.zeros((3,2))
  for i in range(n_imgs_begin, n_imgs_end):
    # Read depth image and camera pose
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % (i))  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.01)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()

  list_pose = []
  i=0
  for iter in range(n_imgs_begin, n_imgs_end):
    print("Fusing frame %d/%d"%(i+1, n_imgs_end-n_imgs_begin))
    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg"%(iter)), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(iter),-1).astype(float)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0

    # Calculate camera pose with ICP algorithm
    if i == 0:
      first_Depthmap = depth_im
      first_Points3D = PointCloud(first_Depthmap, np.linalg.inv(cam_intr))

      pose = np.eye(4)
      list_pose.append(pose)

    elif i == 1:
      second_Depthmap = depth_im
      second_Points3D = PointCloud(second_Depthmap, np.linalg.inv(cam_intr))
      pose, distances, _ = icp(second_Points3D.T, first_Points3D.T) # A, B // maps A onto B : B = pose*A
      pose = np.dot(list_pose[i-1], pose)
      list_pose.append(pose)
      first_Points3D = second_Points3D

    elif i == 2:
      second_Depthmap = depth_im
      second_Points3D = PointCloud(second_Depthmap, np.linalg.inv(cam_intr))
      pose, distances, _ = icp(second_Points3D.T, first_Points3D.T) # A, B // maps A onto B : B = pose*A
      pose = np.dot(list_pose[i-1], pose)
      list_pose.append(pose)

    elif i > 2:
      second_Depthmap = depth_im
      second_Points3D = PointCloud(second_Depthmap, np.linalg.inv(cam_intr))

      # t-1, t-2 프레임들로 얻은 pointcloud로 icp하기
      for j in [3, 2, 1]:
        color_seq = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg" % (iter-j)), cv2.COLOR_BGR2RGB)
        depth_seq = cv2.imread("data/frame-%06d.depth.png" % (iter-j), -1).astype(float)
        depth_seq /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_seq[depth_seq == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)

        # Read depth image and camera pose : t-2
        if j == 3:
          depth_seq_3 = depth_seq
          points_3 = PointCloud(depth_seq_3, np.linalg.inv(cam_intr))
          pose_3 = list_pose[i-3]
          pose_seq = pose_3

          # Create TSDF volume
          view_frust_pts = fusion.get_view_frustum(depth_seq, cam_intr, pose_seq)
          vol_bnds = np.zeros((3, 2))
          vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
          vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
          tsdf_vol_seq = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

        # Read depth image and camera pose : t-2
        elif j == 2:
          depth_seq_2 = depth_seq
          points_2 = PointCloud(depth_seq_2, np.linalg.inv(cam_intr))
          pose_2, distances, _ = icp(points_2.T, points_3.T)  # A, B // maps A onto B : B = pose*A
          pose_seq = np.dot(pose_3, pose_2)

        # Read depth image and camera pose : t-1
        elif j == 1:
          depth_seq_1 = depth_seq
          points_1 = PointCloud(depth_seq_1, np.linalg.inv(cam_intr))
          pose_1, distances, _ = icp(points_1.T, points_2.T)  # A, B // maps A onto B : B = pose*A
          pose_seq = np.dot(pose_2, pose_1)

        tsdf_vol_seq.integrate(color_seq, depth_seq, cam_intr, pose_seq, obs_weight=1.)
      first_Points3D = tsdf_vol_seq.get_point_cloud()

      pose, distances, _ = icp(second_Points3D.T, first_Points3D[0:second_Points3D.shape[1], 0:3]) # A, B // maps A onto B : B = pose*A
      pose = np.dot(list_pose[i-1], pose)

      cam_pose = pose
      list_pose.append(cam_pose)

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, list_pose[i], obs_weight=1.)
    print(f'frame{i+1} \n{list_pose[i]}')
    i=i+1

  fps = (n_imgs_end-n_imgs_begin) / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)