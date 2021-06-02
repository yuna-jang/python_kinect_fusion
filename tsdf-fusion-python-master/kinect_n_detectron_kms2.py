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
from icp_modules.ICP_point_to_plane import *
from icp_modules.ICP_kms import *
from icp_modules.FrameMaps_kms2 import *
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


def Joint_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor


def filter_human(output):
    classes = output["instances"].pred_classes
    human = list(np.nonzero(np.where(classes.numpy() == 0, 1, 0))[0])
    boxes = output["instances"].pred_boxes
    focus = boxes.area().numpy()[human].argmax()
    mask = output["instances"].pred_masks[human[focus]]
    x, y = np.nonzero(1 - mask.numpy())
    return x, y


def filter_joint(output):
    joints = output["instances"].pred_keypoints[0][:, :2].numpy()
    return joints


def joint_to_3D(joints, Inverse, depth_im):
    Joints = np.zeros((3, 17))
    for i in range(17):
        xx, yy = joints[i]
        d = depth_im[int(round(yy)), int(round(xx))]
        Joints[:, i] = d * np.dot(Inverse, np.array([xx, yy, 1]).T)
    return Joints


def simple_bundle(joint_3D: list):
    # Outlier 날리고 평균 구하기.
    joints_3D = np.array(joint_3D)  # N * 3 * 17
    joint_val = [[0, 0, 0] for _ in range(17)]
    counts = [[0.1, 0.1, 0.1] for _ in range(17)]
    mean = np.mean(joints_3D, axis=0)  # 3x17
    std = np.std(joints_3D, axis=0)  # 3x17
    thr_low = mean + 2 * std
    thr_high = mean - 2 * std
    for i in range(len(joints_3D)):
        for j in range(17):
            if thr_low[0, j] < joints_3D[i, 0, j] < thr_high[0, j]:
                joint_val[j][0] += joints_3D[i, 0, j]
                counts[j][0] += 1
            if thr_low[1, j] < joints_3D[i, 1, j] < thr_high[1, j]:
                joint_val[j][1] += joints_3D[i, 1, j]
                counts[j][1] += 1
            if thr_low[2, j] < joints_3D[i, 2, j] < thr_high[2, j]:
                joint_val[j][2] += joints_3D[i, 2, j]
                counts[j][2] += 1
    result = np.zeros((3, 17))
    for i, (val, count) in enumerate(zip(joint_val, counts)):
        xv = val[0]
        xc = count[0]
        yv = val[1]
        yc = count[1]
        zv = val[2]
        zc = count[2]
        result[:, i] = xv / xc, yv / yc, zv / zc
    return result


if __name__ == "__main__":
    seg_model = SEG_model()
    joint_model = Joint_model()

    # Open kinect camera by realtime
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            color_format=pyk4a.ImageFormat.COLOR_MJPG
        )
    )

    # Load video file
    filename = r'C:\Users\82106\PycharmProjects\dino_lib\python_kinect_fusion\tsdf-fusion-python-master\human6.mkv'
    # filename = r'C:\Users\82106\PycharmProjects\dino_lib\python_kinect_fusion\tsdf-fusion-python-master\0531_2.mkv'
    n_frames = 3

    k4a = PyK4APlayback(filename)
    k4a.open()

    # Load Kinect's intrinsic parameter 3X3
    cam_intr = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
    invK = np.linalg.inv(cam_intr)
    # List 생성
    list_depth_im = []
    list_color_im = []
    # vol_bnds 생성
    vol_bnds = np.zeros((3, 2))
    voxel_size = 0.05
    iter = 0
    poses = []
    # while True:
    for i in range(0, n_frames):
        capture = k4a.get_next_capture()
        if capture.depth is not None and capture.color is not None:
            print(f"==========={i}==========")
            # Read depth and color image
            depth_im = capture.transformed_depth.astype(float)
            depth_im /= 1000.
            depth_im[depth_im == 65.535] = 0
            color_capture = convert_to_bgra_if_required(k4a.configuration["color_format"], capture.color)
            color_im = cv2.cvtColor(color_capture, cv2.COLOR_BGR2RGB)
            H, W, d_ = color_im.shape
            list_depth_im.append(depth_im)
            list_color_im.append(color_im)

            if i == 0:
                first_Points3D, sample = PointCloud(depth_im, invK)  # Nx3
                cam_pose = np.eye(4)
                first_pose = cam_pose
                prev_normal = NormalMap(sample, depth_im, invK)  # Normal map이 destination의 normal map 이어야함.

            elif i >= 1:
                second_Points3D, sample = PointCloud(depth_im, invK)  # Nx3
                pose = point_to_plane(second_Points3D,
                                      first_Points3D, prev_normal)  # A, B // maps A onto B : B = pose*A
                prev_normal = NormalMap(sample, depth_im, invK)

                # ## visualize pose result
                # fig = plt.figure(figsize=(8, 8))
                # ax = fig.add_subplot(projection='3d')  # Axe3D object
                # P = np.vstack((second_Points3D.T, np.ones((1, second_Points3D.shape[0]))))  # projection  P = 4XN
                # # ax.scatter(second_Points3D.T[:, 0], second_Points3D.T[:, 1], second_Points3D.T[:, 2], color='g', s=0.5)
                # proj = pose.dot(P)
                # ax.scatter(P.T[:, 0], P.T[:, 1], P.T[:, 2], color='r', s=0.3)
                # ax.scatter(first_Points3D[:, 0], first_Points3D[:, 1], first_Points3D[:, 2], color='b', s=0.3) # fP = Nx3
                # plt.show()
                cam_pose = np.dot(first_pose, pose)

                first_pose = cam_pose
                first_Points3D = second_Points3D

                # Compute camera view frustum and extend convex hull
                view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
            poses.append(cam_pose)
            i = i + 1
    # ======================================================================================================== #
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)
    k4a.close()

    # ===============Integrate===============
    n_imgs = len(list_depth_im)
    iter = 0
    joints_3D = []

    for i in range(0, n_imgs):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))

        # Read depth and color image
        depth_im = list_depth_im[iter]
        color_im = list_color_im[iter]

        output = seg_model(color_im)
        not_valid_x, not_valid_y = filter_human(output)
        for not_x, not_y in zip(not_valid_x, not_valid_y):
            depth_im[not_x, not_y] = 0
            color_im[not_x, not_y] = 0
        val_x, val_y = np.nonzero(depth_im)
        threshold = np.mean(depth_im[val_x, val_y]) + 2 * np.std(depth_im[val_x, val_y])
        depth_im[depth_im >= threshold] = 0
        H, W = depth_im.shape

        output = joint_model(color_im)
        joint = filter_joint(output)
        joints_3D.append(joint_to_3D(joint, invK, depth_im))

        # # Set first frame as world system
        # if iter == 0:
        #     previous_Points3D, _ = PointCloud(depth_im, invK)
        #     cam_pose = np.eye(4)
        #     previous_pose = cam_pose
        #     prev_normal = NormalMap(sample, depth_im, invK)
        #
        # elif iter == 1:
        #     second_Points3D, sample = PointCloud(depth_im, invK)
        #     pose = point_to_plane(second_Points3D,
        #                           first_Points3D, prev_normal)  # A, B // maps A onto B : B = pose*A
        #     prev_normal = NormalMap(sample, depth_im, invK)
        #     cam_pose = np.dot(previous_pose, pose)
        #     previous_pose = cam_pose
        #     previous_Points3D = second_Points3D
        #
        # elif iter > 1:
        #     Points3D, sample = PointCloud(depth_im, invK)
        #     # Compute camera view frustum and extend convex hull
        #     pose = point_to_plane(Points3D, previous_Points3D, prev_normal)  # A, B // maps A onto B : B = pose*A
        #     prev_normal = NormalMap(sample, depth_im, invK)
        #     pose = np.dot(previous_pose, pose)
        #     view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, pose)
        #     vol_bnds_seq = np.zeros((3, 2))
        #     vol_bnds_seq[:, 0] = np.minimum(vol_bnds_seq[:, 0], np.amin(view_frust_pts, axis=1))
        #     vol_bnds_seq[:, 1] = np.maximum(vol_bnds_seq[:, 1], np.amax(view_frust_pts, axis=1))
        #     tsdf_vol_seq = fusion.TSDFVolume(vol_bnds_seq, voxel_size=voxel_size)
        #     tsdf_vol_seq.integrate(color_im, depth_im, cam_intr, pose, obs_weight=1.)
        #     # second_Points3D = tsdf_vol_seq.get_point_cloud()[:, 0:3]
        #     #
        #     # # 누적 pointcloud vertex only
        #     # first_Points3D = tsdf_vol.get_partial_point_cloud()
        #     #
        #     # pts_size = min(first_Points3D.shape[0], second_Points3D.shape[0])
        #     # pose = point_to_plane(second_Points3D[0:pts_size, :],
        #     #                          first_Points3D[0:pts_size, :], normal_map)  # A, B // maps A onto B : B = pose*A
        #     # print(f'{pts_size} / {first_Points3D.shape[0]}')
        #     pose = np.dot(previous_pose, pose)
        #
        #     cam_pose = pose
        #     previous_pose = cam_pose
        #     previous_Points3D = Points3D
        # poses.append(previous_pose)
        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_im, depth_im, cam_intr, poses[i], obs_weight=1.)
        i += 1

    joint_ = simple_bundle(joints_3D)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')  # Axe3D object
    print(joint_.shape)
    print(joint_)
    for i in range(17):
        ax.scatter(joint_[0, i], joint_[1, i], joint_[2, i]) # projection  P = 4XN
    plt.show()

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh")
    # verts, faces, norms, colors = human_vol.get_mesh()
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("human_mesh.ply", verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    # print("Saving point cloud")
    # point_cloud = human_vol.get_point_cloud()
    # fusion.pcwrite("human_pcd.ply", point_cloud)
