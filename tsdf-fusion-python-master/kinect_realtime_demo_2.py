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
  n_imgs = 50
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  vol_bnds = np.zeros((3,2))
  for i in range(n_imgs):
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
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection='3d')


  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()

  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection='3d')

  for i in range(n_imgs):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg"%(i)), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0

    # Set first frame as world system
    if i == 0:
      first_Depthmap = depth_im
      first_Points3D = PointCloud(first_Depthmap, np.linalg.inv(cam_intr))
      cam_pose = np.eye(4)
      first_pose = cam_pose
    elif i == 1:
      second_Depthmap = depth_im
      second_Points3D = PointCloud(second_Depthmap, np.linalg.inv(cam_intr))
      pose, distances, _ = icp(second_Points3D.T, first_Points3D.T) # A, B // maps A onto B : B = pose*A
      pose = np.dot(first_pose, pose)
      print(f'frame{i} \n{pose}')

      cam_pose = pose
      first_pose = cam_pose

    elif i>1:
      second_Depthmap = depth_im
      second_Points3D = PointCloud(second_Depthmap, np.linalg.inv(cam_intr))

      # t-1, t-2 프레임들로 얻은 pointcloud로 icp하기
      # Read depth image and camera pose : t-2
      depth_im = cv2.imread("data/frame-%06d.depth.png" % (i-2), -1).astype(float)
      depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
      depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
      # pose
      depthmap_2 = depth_im
      points_2 = PointCloud(depthmap_2, np.linalg.inv(cam_intr))
      pose_2 = np.eye(4)

      # Compute camera view frustum and extend convex hull
      view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, pose_2)
      vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
      vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
      pre_tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

      # Read depth image and camera pose : t-1
      color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg" % (i-1)), cv2.COLOR_BGR2RGB)
      depth_im = cv2.imread("data/frame-%06d.depth.png" % (i), -1).astype(float)
      depth_im /= 1000.
      depth_im[depth_im == 65.535] = 0
      # pose
      depthmap_1 = depth_im
      points_1 = PointCloud(depthmap_1, np.linalg.inv(cam_intr))
      pose_1, distances, _ = icp(points_1.T, points_2.T)  # A, B // maps A onto B : B = pose*A

      pre_tsdf_vol.integrate(color_image, depth_im, cam_intr, pose_1, obs_weight=1.)
      first_Points3D = tsdf_vol.get_point_cloud()

      # pose, distances, _ = icp(second_Points3D.T, first_Points3D.T) # A, B // maps A onto B : B = pose*A
      pose, distances, _ = icp(second_Points3D.T, first_Points3D[0:second_Points3D.shape[1], 0:3]) # A, B // maps A onto B : B = pose*A



      # # knn으로 n개 뽑기 nx3
      # first_Points3D = tsdf_vol.get_point_cloud()[:, 0:3]
      # # indices는 dst꺼
      # dist, indices = nearest_neighbor(second_Points3D.T, first_Points3D)
      # # dist, indices = nearest_neighbor(first_Points3D, second_Points3D.T)
      # zipped = zip(dist, indices)
      # zipped = list(zipped)
      # res = sorted(zipped, key=lambda x: x[1])
      # ind = []
      # for j in range(second_Points3D.shape[1]):
      #   ind.append(res[j][1])
      #
      # ax.scatter(second_Points3D[0, :], second_Points3D[1, :], second_Points3D[2, :], c='g', s=0.1)
      # ax.scatter(first_Points3D[ind,0],first_Points3D[ind,1], first_Points3D[ind,2], c='r', s=0.1)
      # plt.show()
      #
      # pose, distances, _ = icp(second_Points3D.T, first_Points3D[ind,:])  # A, B // maps A onto B : B = pose*A


      pose = np.dot(first_pose, pose)
      print(f'frame{i} \n{pose}')

      cam_pose = pose
      first_pose = cam_pose

      # #icp 검증
      # if i == 10:
      #   P = np.vstack((first_Points3D, np.ones((1, first_Points3D.shape[1]))))
      #   P_projection = pose.dot(P)
      #   ax.scatter(P_projection[0, :], P_projection[1, :], P_projection[2, :], c='g', s=0.1)
      #
      # if i == 50:
      #   Q = np.vstack((first_Points3D, np.ones((1, first_Points3D.shape[1]))))
      #   Q_projection = pose.dot(Q)
      #   ax.scatter(Q_projection[0, :], Q_projection[1, :], Q_projection[2, :], c='r', s=0.1)
      #   plt.show()

    # Integrate observation into voxel volume (assume color aligned with depth)
    print(depth_im.shape)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    if i == 7:
      break

  # plt.show()

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)