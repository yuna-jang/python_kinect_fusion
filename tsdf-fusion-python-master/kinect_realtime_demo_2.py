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
  n_imgs = 100
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
      cam_pose = np.matrix('9.093128999999999795e-01 2.726222899999999894e-01 -3.142243299999999961e-01 -3.404563400000000239e-01;'
                           '-2.724861800000000223e-01 9.610497999999999541e-01 4.527962600000000337e-02 1.646981800000000065e-02;'
                           '3.143392500000000145e-01 4.444964600000000238e-02 9.482093499999999509e-01 2.965691699999999931e-01;'
                           '0.000000000000000000e+00 0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00')
      first_pose = cam_pose
    else:
      # second_Depthmap = depth_im
      # second_Points3D = PointCloud(second_Depthmap, np.linalg.inv(cam_intr))
      second_Points3D = tsdf_vol.get_point_cloud()[0:first_Points3D.shape[1], 0:3]

      pose, distances, _ = icp(second_Points3D, first_Points3D.T)  # A, B // maps A onto B : B = pose*A

      pose = np.dot(first_pose, pose)
      print(f'frame{i} \n{pose}')

      cam_pose = pose
      first_Points3D = second_Points3D.T
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