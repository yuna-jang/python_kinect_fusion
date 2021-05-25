import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)

import fusion

# Kinect module
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

  ## Open kinect camera by realtime
  k4a = PyK4A(
    Config(
      color_resolution=pyk4a.ColorResolution.RES_720P,
      depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
      synchronized_images_only=True,
    )
  )
  try:
     k4a.start()
     print("키넥트 모드")
  except:
    filename = r'0_sample_video\moving2.mkv'
    k4a = PyK4APlayback(filename)
    k4a.open()
    print("비디오 모드")


  ## Load Kinect's intrinsic parameter
  intrinsic_color = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
  intrinsic_depth = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.DEPTH)
  distortion = k4a.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR)
  cam_intr = intrinsic_color
  print(cam_intr)

  # vol_bnds 생성
  vol_bnds = np.zeros((3, 2))

  t0_elapse = time.time() # for check fps
  iter = 0
  while True:
    try:
      capture = k4a.get_capture()
    except:
        capture = k4a.get_next_capture()
    if capture.depth is not None and capture.color is not None:
      iter = iter + 1
      print(f"==========={iter}==========")

      # Read depth and color image
      depth_im = capture.transformed_depth.astype(float)
      depth_im /= 1000.  ## depth is saved in 16-bit PNG in millimeters
      depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset) 65.535=2^16/1000
      color_image = cv2.cvtColor(capture.color, cv2.COLOR_BGR2RGB)

      # Show
      # cv2.imshow("Depth", colorize(capture.transformed_depth, (None, 5000)))
      # cv2.imshow("Color", color_image)


      # Set first frame as world system
      if iter == 1:
        first_Depthmap = depth_im
        first_Points3D = PointCloud(first_Depthmap, np.linalg.inv(cam_intr))
        cam_pose = np.eye(4)
        first_pose = cam_pose
      else:
        # legacy
        # second_Depthmap = depth_im
        # second_Points3D = PointCloud(second_Depthmap, np.linalg.inv(cam_intr))

        second_Points3D = tsdf_vol.get_point_cloud()[0:first_Points3D.shape[1], 0:3]

        pose, distances, _ = icp(second_Points3D.T, first_Points3D) # A, B // maps A onto B : B = pose*A
        pose = np.dot(first_pose,pose)

        cam_pose = pose
        first_Points3D = second_Points3D
        first_pose = cam_pose

      # Compute camera view frustum and extend convex hull
      view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
      vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
      vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

      # ======================================================================================================== #

      # Initialize voxel volume
      if iter == 1:
        print("Initializing voxel volume...")
        tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.01)
      else:
        tsdf_vol.set_vol_bnds(vol_bnds)


      # Loop through RGB-D images and fuse them together
      print("Fusing frame")

      # Integrate observation into voxel volume (assume color aligned with depth)
      tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)


      if iter==200:
          break


      key = cv2.waitKey(10)
      if key != -1:
        cv2.destroyAllWindows()
        break


    key = cv2.waitKey(10)
    if key != -1:
      cv2.destroyAllWindows()
      break

  try:
    k4a.stop()
  except:
    k4a.close()

  fps = iter / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to test_mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("test_mesh_sample.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to test_pcd.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("test_pcd.ply", point_cloud)