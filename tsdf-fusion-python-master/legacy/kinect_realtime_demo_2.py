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
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.03)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()


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
      cam_pose = pose
      first_pose = cam_pose

    elif i > 1:
      second_Depthmap = depth_im
      second_Points3D = PointCloud(second_Depthmap, np.linalg.inv(cam_intr))

      # Compute camera view frustum and extend convex hull
      view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
      vol_bnds = np.zeros((3, 2))
      vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
      vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
      tsdf_vol_seq = fusion.TSDFVolume(vol_bnds, voxel_size=0.03)
      tsdf_vol_seq.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
      second_Points3D = tsdf_vol_seq.get_point_cloud()[:, 0:3]

      # # knn으로 n개 뽑기 nx3
      first_Points3D = tsdf_vol.get_point_cloud()[:, 0:3]
      # src의 개수만큼 dst의 indices를 리턴
      neigh = NearestNeighbors(n_neighbors=1)
      # 이론상 이거
      neigh.fit(first_Points3D)  # dst
      distances, indices = neigh.kneighbors(second_Points3D, return_distance=True)  # src

      # neigh.fit(second_Points3D.T) # dst
      # distances, indices = neigh.kneighbors(first_Points3D, return_distance=True) # src


      zipped = zip(distances, indices)
      zipped = list(zipped)
      res = sorted(zipped, key=lambda x: np.mean(x[0]))
      ind = []
      num_points = second_Points3D.shape[0]

      # num_points개가 안나옴
      for j in range(0,len(res)):
        for k in range(0, len(res[j][1])):
          if not(res[j][1][k] in ind):
            ind.append(res[j][1][k])
            if(len(ind) == num_points):
              break

      samples = random.sample(range(first_Points3D.shape[0]), num_points-len(ind))
      ind_set = ind + samples

      pts_size = min(first_Points3D.shape[0], second_Points3D.shape[0])
      pose, distances, _ = icp(second_Points3D[0:pts_size,:], first_Points3D[0:pts_size,:])  # A, B // maps A onto B : B = pose*A
      pose = np.dot(first_pose, pose)

      # pose matrix 검증
      fig = plt.figure(figsize=(8, 8))
      ax = fig.add_subplot(projection='3d')  # Axe3D object
      P = np.vstack((second_Points3D.T, np.ones((1, second_Points3D.T.shape[1])))) # projection
      proj = pose.dot(P)
      # ax.scatter(P.T[:, 0], P.T[:, 1], P.T[:, 2], color='r', s=0.3)
      ax.scatter(proj.T[:, 0], proj.T[:, 1], proj.T[:, 2], color='g', s=0.3)
      ax.scatter(first_Points3D[:, 0], first_Points3D[:, 1], first_Points3D[:, 2], color='b', s=0.3)
      plt.show()

      cam_pose = pose
      first_pose = cam_pose

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
    # # pose matrix 검증
    # if i == 1:
    #   fig = plt.figure(figsize=(8, 8))
    #   ax = fig.add_subplot(projection='3d')  # Axe3D object
    #   P = np.vstack((second_Points3D, np.ones((1, second_Points3D.shape[1]))))  # projection
    #   # ax.scatter(second_Points3D.T[:, 0], second_Points3D.T[:, 1], second_Points3D.T[:, 2], color='g', s=0.5)
    #   proj = pose.dot(P)
    #   ax.scatter(P.T[:, 0], P.T[:, 1], P.T[:, 2], color='r', s=0.3)
    #   ax.scatter(first_Points3D.T[:, 0], first_Points3D.T[:, 1], first_Points3D.T[:, 2], color='b', s=0.3)
    #   pocla = tsdf_vol.get_point_cloud()[:, 0:3] # nx3
    #   ax.scatter(pocla[:, 0], pocla[:, 1], pocla[:, 2], color='g', s=0.3)
    #   plt.show()

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