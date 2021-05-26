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
  n_imgs_end = 810
  n_imgs = n_imgs_end - n_imgs_begin
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
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()

  i = 0
  for iter in range(n_imgs_begin, n_imgs_end):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg"%(iter)), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(iter),-1).astype(float)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0

    # Set first frame as world system
    if i == 0:
      first_Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))
      cam_pose = np.eye(4)
      first_pose = cam_pose

    else:
      second_Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))

      ind = random.sample(range(first_Points3D.shape[1]), second_Points3D.shape[1])
      pose, distances, _ = icp(second_Points3D.T, first_Points3D.T[ind,:])  # A, B // maps A onto B : B = pose*A
      cam_pose = np.dot(first_pose, pose)

      first_pose = cam_pose

      # pointclouds
      P = np.vstack((second_Points3D, np.ones((1, second_Points3D.shape[1]))))  # projection
      proj = cam_pose.dot(P)[0:3,:]
      first_Points3D = np.hstack((first_Points3D, proj))

      # pose matrix 검증
      fig = plt.figure(figsize=(8, 8))
      ax = fig.add_subplot(projection='3d')  # Axe3D object
      ax.scatter(second_Points3D[0,:], second_Points3D[1,:], second_Points3D[2,:], color='r', s=0.3)  # projection 전의 위치
      ax.scatter(proj[0, :], proj[1,:], proj[2,:], color='g', s=0.3)  # icp로 얻은 pose로 projection한 pointcloud
      ax.scatter(first_Points3D[0,:], first_Points3D[1,:], first_Points3D[2,:], color='b', s=0.3)  # 누적 pointcloud 전체
      plt.show()

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
    i=i+1


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