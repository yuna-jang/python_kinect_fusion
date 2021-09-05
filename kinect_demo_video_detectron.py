import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

np.set_printoptions(threshold=np.inf)
import fusion
# Kinect module
import pyk4a
from helpers import convert_to_bgra_if_required
from pyk4a import Config, PyK4A
from pyk4a import PyK4APlayback
from icp_modules.ICP_algorithm import *
from icp_modules.FramePreprocessing import PointCloud
from helpers import colorize, convert_to_bgra_if_required
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def SEG_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    """
    class 중 label 0 이 human
    pred_masks : mask prediction(segmentation)
    """
    return predictor


def filter_human(output):
    classes = output["instances"].pred_classes
    human = list(np.nonzero(np.where(classes.numpy() == 0, 1, 0))[0])
    boxes = output["instances"].pred_boxes
    focus = boxes.area().numpy()[human].argmax()
    mask = output["instances"].pred_masks[human[focus]]
    x, y = np.nonzero(1 - mask.numpy())
    return x, y


if __name__ == "__main__":
    model = SEG_model()

    # Open kinect camera by realtime
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            color_format=pyk4a.ImageFormat.COLOR_MJPG
        )
    )

    # Load video file
    filename = r'0_sample_video\yuna2.mkv'

    n_frames = 20

    k4a = PyK4APlayback(filename)
    k4a.open()

    # Load Kinect's intrinsic parameter
    cam_intr = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)

    # List 생성
    list_depth_im = []
    list_color_im = []
    # vol_bnds 생성
    vol_bnds = np.zeros((3, 2))
    voxel_size = 0.01
    iter = 0

    # for k in range(100):
    #     k4a.get_next_capture()
    for i in range(0, n_frames):
        capture = k4a.get_next_capture()
        if capture.depth is not None and capture.color is not None: #and i%2==0:
            print(f"==========={iter}==========")
            # Read depth and color image
            depth_im = capture.transformed_depth.astype(float)
            depth_im /= 1000.  ## depth is saved in 16-bit PNG in millimeters
            depth_im[depth_im >= 2] = 0  # set invalid depth to 0 (specific to 7-scenes dataset) 65.535=2^16/1000
            color_capture = convert_to_bgra_if_required(k4a.configuration["color_format"], capture.color)
            color_im = cv2.cvtColor(color_capture, cv2.COLOR_BGR2RGB)

            filtered_depth = cv2.bilateralFilter(depth_im.astype(np.float32), 5, 35, 35)
            color_im = cv2.bilateralFilter(color_im, 10, 50, 50)
            sharpening_2 = np.array([[-1, -1, -1, -1, -1],
                                     [-1, 2, 2, 2, -1],
                                     [-1, 2, 9, 2, -1],
                                     [-1, 2, 2, 2, -1],
                                     [-1, -1, -1, -1, -1]]) / 9.0
            color_im = cv2.filter2D(color_im, -1, sharpening_2)

            # Segmentaion human
            output = model(color_im)
            not_valid_x, not_valid_y = filter_human(output)
            for not_x, not_y in zip(not_valid_x, not_valid_y):
                depth_im[not_x, not_y] = 0
                color_im[not_x, not_y] = 0

            list_depth_im.append(depth_im)
            list_color_im.append(color_im)

            if iter == 0:
                first_Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))
                cam_pose = np.eye(4)
                first_pose = cam_pose

            elif iter >= 1:
                second_Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))
                ind = random.sample(range(first_Points3D.shape[1]), second_Points3D.shape[1])
                pose, _ = icp(second_Points3D.T,first_Points3D.T[ind, :])  # A, B // maps A onto B : B = pose*A
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
    # human_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)
    k4a.close()
    poses = []
    # ===============Integrate===============
    n_imgs = len(list_depth_im)
    iter = 0
    for iter in range(0, n_imgs):
        print("Fusing frame %d/%d" % (iter + 1, n_imgs))

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
            pose, _ = icp(Points3D.T, previous_Points3D.T)  # A, B // maps A onto B : B = pose*A
            pose = np.dot(previous_pose, pose)

            view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, pose)
            vol_bnds_seq = np.zeros((3, 2))
            vol_bnds_seq[:, 0] = np.minimum(vol_bnds_seq[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds_seq[:, 1] = np.maximum(vol_bnds_seq[:, 1], np.amax(view_frust_pts, axis=1))
            tsdf_vol_seq = fusion.TSDFVolume(vol_bnds_seq, voxel_size=voxel_size)
            tsdf_vol_seq.integrate(color_im, depth_im, cam_intr, pose, obs_weight=1.)
            second_Points3D = tsdf_vol_seq.get_point_cloud()[:, 0:3]

            # 누적 pointcloud vertex only
            first_Points3D = tsdf_vol.get_point_cloud()[:, 0:3]

            pts_size = min(first_Points3D.shape[0], second_Points3D.shape[0])
            size = min(pts_size, 30000)

            samples_first = random.sample(range(first_Points3D.shape[0]), size)
            samples_second = random.sample(range(second_Points3D.shape[0]), size)
            pose_real, mean_error = icp(second_Points3D[samples_second, :],first_Points3D[samples_first, :])  # A, B // maps A onto B : B = pose*A
            print(f'icp stop : {mean_error}')

            pose_real = np.dot(previous_pose, pose_real)

            cam_pose = pose_real
            previous_pose = cam_pose
            previous_Points3D = Points3D

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_im, depth_im, cam_intr, cam_pose, obs_weight=1.)
        iter = iter + 1


    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    # verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("human_mesh_seg.ply", verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite("human_pcd_seg.ply", point_cloud)
