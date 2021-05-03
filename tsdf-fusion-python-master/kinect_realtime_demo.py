"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion

# Kinect module
import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A

from icp_modules.ICP import ICP_point_to_point
from icp_modules.FramePreprocessing import DepthMap,PointCloud

if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")

  ## Open kinect camera by realtime
  k4a = PyK4A(
    Config(
      color_resolution=pyk4a.ColorResolution.RES_720P,
      depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
    )
  )
  k4a.start()

  ## Load Kinect's intrinsic parameter
  intrinsic_color = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
  intrinsic_depth = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.DEPTH)
  distortion = k4a.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR)
  cam_intr = intrinsic_color

  # vol_bnds 생성
  vol_bnds = np.zeros((3, 2))

  t0_elapse = time.time() # for check fps
  iter = 0
  while True:
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    capture = k4a.get_capture()
    if capture.depth is not None:
      iter = iter + 1
      # Read depth image and camera pose
      cv2.imshow("Depth", colorize(capture.depth, (None, 5000)))
      cv2.imshow("Color", capture.transformed_color)

      depth_im = capture.depth.astype(float)
      depth_im /= 1000.  ## depth is saved in 16-bit PNG in millimeters
      depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset) 65.535=2^16/1000

      # KinectFusion에서 pose estimation하는 것을 참고하여 연산

      # Set first frame as world system
      if iter == 1:
        first_Depthmap = depth_im
        first_Points3D = PointCloud(first_Depthmap, np.linalg.inv(cam_intr))
        continue
      elif iter == 2:
        second_Depthmap = depth_im
        second_Points3D = PointCloud(second_Depthmap, np.linalg.inv(cam_intr))
        # ICP find Trans between neighboring frames
        first_pose = ICP_point_to_point(first_Points3D, second_Points3D)
        first_Points3D = second_Points3D
        continue
      else:
        second_Depthmap = depth_im
        second_Points3D = PointCloud(second_Depthmap, np.linalg.inv(cam_intr))
        # ICP find Trans between neighboring frames
        pose = ICP_point_to_point(first_Points3D, second_Points3D)
        cam_pose = np.dot(first_pose, pose)
        first_Points3D = second_Points3D
        first_pose = cam_pose

      cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % (3))  # 4x4 rigid transformation matrix


      # Compute camera view frustum and extend convex hull
      view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
      vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
      vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
      # ======================================================================================================== #

      # ======================================================================================================== #
      # Integrate
      # ======================================================================================================== #
      # Initialize voxel volume
      print("Initializing voxel volume...")
      if iter == 3:
        tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.01)
      else:
        tsdf_vol.set_vol_bnds(vol_bnds)

      # Loop through RGB-D images and fuse them together
      print("Fusing frame")

      # Read RGB-D image and camera pose
      color_image = cv2.cvtColor(capture.transformed_color,cv2.COLOR_BGR2RGB)
      depth_im =capture.depth.astype(float)
      depth_im /= 1000.  ## depth is saved in 16-bit PNG in millimeters
      depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset) 65.535=2^16/1000

      # Integrate observation into voxel volume (assume color aligned with depth)
      tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
      print(f"==========={iter}==========")

    if iter==100:
      break
    key = cv2.waitKey(10)
    if key != -1:
      cv2.destroyAllWindows()
      break

  k4a.stop()

  fps = iter / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to test_mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("test_mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to test_pcd.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("test_pcd.ply", point_cloud)