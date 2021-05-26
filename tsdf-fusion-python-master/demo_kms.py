"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import fusion

import pyk4a
from helpers import colorize
from pyk4a import Config, PyK4A
from pyk4a import PyK4APlayback

from icp_modules.ICP_kms import *
from icp_modules.FramePreprocessing import PointCloud

from helpers import colorize, convert_to_bgra_if_required


class Volumes:
    def __init__(self, num, vol_bnds, voxel_size):
        self.vsize = voxel_size
        self.vol_bnd = vol_bnds
        self.fusions = [fusion.TSDFVolume(self.vol_bnd, self.vsize)
                        for _ in range(num)]
        self.cloud_ICP = None

    def Fuse(self, color_image, depth_im, cam_intr, cam_pose, index):
        if index + 1 < len(self.fusions):
            for idx in range(index+1):
                Fusion = self.fusions[idx]
                Fusion.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
                self.fusions[idx] = Fusion
            self.cloud_ICP = self.fusions[0]
        else:
            self.cloud_ICP = self.fusions.pop(0)
            for idx in range(len(self.fusions)):
                Fusion = self.fusions[idx]
                Fusion.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
                self.fusions[idx] = Fusion
            self.fusions.append(fusion.TSDFVolume(self.vol_bnd, self.vsize))

    def get_cloud_ICP(self):
        cloud = self.cloud_ICP.get_point_cloud()[:, 0:3]
        print("cloud for ICP: ", cloud.shape)
        return cloud





if __name__ == "__main__":
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    print("Estimating voxel volume bounds...")
    n_imgs = 80
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
    vol_bnds = np.zeros((3, 2))
    for i in range(n_imgs):
        # Read depth image and camera pose
        depth_im = cv2.imread("data/frame-%06d.depth.png" % (i), -1).astype(float)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % (i))  # 4x4 rigid transformation matrix

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
    voxel_size = 0.02
    # volumes = Volumes(3, vol_bnds, voxel_size=voxel_size)
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)


    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()

    for i in range(n_imgs):

        print("Fusing frame %d/%d" % (i + 1, n_imgs))

        # Read RGB-D image(480, 640) and camera pose
        color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg" % (i)), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread("data/frame-%06d.depth.png" % (i), -1).astype(float)
        GT_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0

        # Set first frame as world system
        if i == 0:
            next_Depthmap = depth_im
            next_Points3D = PointCloud(next_Depthmap, np.linalg.inv(cam_intr))
            cam_pose = np.eye(4)
            first_pose = cam_pose
        else:
            prev_Points3D = next_Points3D.copy()
            next_Depthmap = depth_im
            next_Points3D = PointCloud(next_Depthmap, np.linalg.inv(cam_intr))
            pose, distances, _ = icp(next_Points3D.T, prev_Points3D.T)

            pose = np.dot(first_pose, pose)
            print('Ground Truth', GT_pose)
            print('Estimated pose', pose)
            # print(f'frame{i} \n{pose}')

            cam_pose = pose
            first_pose = cam_pose

        # Integrate observation into voxel volume (assume color aligned with depth)
        print(depth_im.shape)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("mesh_kms.ply", verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite("pc.ply", point_cloud)
