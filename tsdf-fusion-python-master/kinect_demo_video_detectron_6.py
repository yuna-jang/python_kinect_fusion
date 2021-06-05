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
from icp_modules.ICP import *
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


def filter_joint(output, color_img, depth_img):
    joints = output["instances"].pred_keypoints[0][:, :2].numpy()    # 17x2
    H, W, C = color_img.shape
    color_filtered = []
    depth_filtered = []
    for i in range(len(joints)):
        joint_xy = joints[i]
        xx = int(np.round(joint_xy[0]))
        yy = int(np.round(joint_xy[1]))
        filter_color = np.zeros((H, W, 3))
        filter_color[yy, xx] = color_img[yy, xx]
        color_filtered.append(filter_color)
        filter_depth = np.zeros((H, W))
        filter_depth[yy, xx] = depth_img[yy, xx]
        depth_filtered.append(filter_depth)
    return joints, color_filtered, depth_filtered


# def joint_TSDF(vols, color_filtered, depth_filtered, K, M):
#     for vol, color, depth in zip(vols, color_filtered, depth_filtered):
#         vol.integrate(color, depth, K, M, obs_weight=1.)
#     return vols


def vol_to_joint(vols):
    result = []
    for i in range(len(vols)):
        joints, _a, _b, _c = vols[i].get_mesh()
        temp = joints.reshape(-1, 3)
        result.append(np.mean(temp, axis=0))
    result = np.array(result)
    print(result.shape)
    return result


def Joint_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor


# def joint_to_3D(joints, Inverse, pose, depth_im):
#     Joints = np.zeros((3, 17))
#     # invPose = np.linalg.inv(pose)
#     for i in range(17):
#         xx, yy = joints[i]
#         d = depth_im[int(round(yy)), int(round(xx))]
#         U = d * np.dot(Inverse, np.array([xx, yy, 1]).T)    # 3X1
#         Joints[:, i] = np.dot(pose, np.array([U[0], U[1], U[2], 1]))[:-1]   # [4x4 @ 4x1][0, 1, 2]
#     return Joints   # 3x17

#
#
# def simple_bundle(joint_3D: list):
#     # Outlier 날리고 평균 구하기.
#     joints_3D = np.array(joint_3D)  # N * 3 * 17
#     if len(joint_3D) > 20:
#         m = np.mean(joints_3D, axis=0)
#         s = np.std(joints_3D, axis=0)
#         thr = m + 2*s
#         thr2 = m - 2*s
#         vals = np.zeros((3, 17))
#         count = np.zeros((3, 17))
#         for i in range(len(joints_3D)):
#             for k in range(3):
#                 for j in range(17):
#                     if thr2[k, j] <= joints_3D[i, k, j] <= thr[k, j]:
#                         vals[k, j] += joints_3D[i, k, j]
#                         count[k, j] += 1
#         result = vals / count
#     else:
#         result = np.mean(joints_3D, axis=0)
#     return result
#
#
# def coordinate_transfer(vol_bnd, joint_val):
#     print('vol_bnd', vol_bnd)
#     vol_max = np.max(vol_bnd, axis=1)
#     vol_min = np.min(vol_bnd, axis=1)
#     joint_max = np.max(vol_bnd, axis=1)
#     joint_min = np.min(vol_bnd, axis=1)
#     print(joint_max, joint_min)
#     for i in range(17):
#         joint_val[:, i] = (joint_val[:, i] - joint_min) / (joint_max - joint_min) * (vol_max - vol_min) + vol_min
#     return joint_val

def panoptic_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor


def human_masking(p_model, color_image, depth_image):
    panoptic_seg, segments_info = p_model(color_image)["panoptic_seg"]
    panop = panoptic_seg.numpy()
    panop[panop != 1] = 0
    panop_3 = np.array([panop, panop, panop])
    panop_3 = panop_3.transpose(1, 2, 0)
    masked_color = np.multiply(color_image, panop_3)
    masked_depth = np.multiply(depth_image, panop)
    return masked_color, masked_depth



if __name__ == "__main__":
    model = SEG_model()
    joint_model = Joint_model()
    panoptic = panoptic_model()
    # Open kinect camera by realtime
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            color_format=pyk4a.ImageFormat.COLOR_MJPG
        )
    )

    # Load video file
    filename = r'C:\Users\82106\PycharmProjects\dino_lib\python_kinect_fusion\tsdf-fusion-python-master\0531_3.mkv'

    n_frames = 6

    k4a = PyK4APlayback(filename)
    k4a.open()

    # Load Kinect's intrinsic parameter
    cam_intr = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)

    # List 생성
    list_depth_im = []
    list_color_im = []
    # vol_bnds 생성
    vol_bnds = np.zeros((3, 2))
    voxel_size = 0.02
    iter = 0
    poses = []
    # while True:

    for i in range(0, n_frames):

        capture = k4a.get_next_capture()
        if capture.depth is not None and capture.color is not None: #and i%2==0:
            print(f"==========={iter}==========")
            # Read depth and color image
            depth_im = capture.transformed_depth.astype(float)
            depth_im /= 1000.  ## depth is saved in 16-bit PNG in millimeters
            depth_im[depth_im >= 1.9] = 0  # set invalid depth to 0 (specific to 7-scenes dataset) 65.535=2^16/1000
            color_capture = convert_to_bgra_if_required(k4a.configuration["color_format"], capture.color)
            color_im = cv2.cvtColor(color_capture, cv2.COLOR_BGR2RGB)

            # cv2.imshow('mask', depth_mask)
            # cv2.waitKey(0)
            filtered_depth = cv2.bilateralFilter(depth_im.astype(np.float32), 5, 35, 35)
            color_im = cv2.bilateralFilter(color_im, 10, 50, 50)
            sharpening_2 = np.array([[-1, -1, -1, -1, -1],
                                     [-1, 2, 2, 2, -1],
                                     [-1, 2, 9, 2, -1],
                                     [-1, 2, 2, 2, -1],
                                     [-1, -1, -1, -1, -1]]) / 9.0
            color_im = cv2.filter2D(color_im, -1, sharpening_2)

            color, depth_im = human_masking(panoptic, color_im, depth_im)
            depth_mask = depth_im.copy()
            depth_mask[depth_mask != 0] = 1
            depth_mask = np.array([depth_mask, depth_mask, depth_mask]).transpose(1, 2, 0)

            color_im = np.multiply(color.astype(np.uint8), depth_mask.astype(np.uint8))
            list_depth_im.append(depth_im)
            list_color_im.append(color_im)

            if iter == 0:
                first_Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))
                cam_pose = np.eye(4)
                first_pose = cam_pose

            elif iter >= 1:
                second_Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))
                ind = random.sample(range(first_Points3D.shape[1]), second_Points3D.shape[1])
                pose, distances, _ = icp(second_Points3D.T,
                                         first_Points3D.T[ind, :])  # A, B // maps A onto B : B = pose*A
                cam_pose = np.dot(first_pose, pose)

                first_pose = cam_pose.copy()
                first_Points3D = second_Points3D.copy()

                # Compute camera view frustum and extend convex hull
                view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
                vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
                vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
            # tsdf_vol.integrate(color_im, depth_im, cam_intr, cam_pose, obs_weight=1.)
            poses.append(cam_pose)
            iter = iter + 1

    # ======================================================================================================== #
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)
    # human_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)
    k4a.close()
    # poses = []
    # ===============Integrate===============
    n_imgs = len(list_depth_im)
    iter = 0
    for iter in range(0, n_imgs):
        print("Fusing frame %d/%d" % (iter + 1, n_imgs))

        # Read depth and color image
        depth_im = list_depth_im[iter]
        color_im = list_color_im[iter]
        pose = poses[iter]
        tsdf_vol.integrate(color_im, depth_im, cam_intr, pose, obs_weight=1.)
        iter = iter + 1
        # Set first frame as world system
        # if iter == 0:
        #     previous_Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))
        #     cam_pose = np.eye(4)
        #     previous_pose = cam_pose
        #
        # elif iter >= 1:
        #     Points3D = PointCloud(depth_im, np.linalg.inv(cam_intr))
        #
        #     # Compute camera view frustum and extend convex hull
        #     pose, distances, _ = icp(Points3D.T, previous_Points3D.T)  # A, B // maps A onto B : B = pose*A
        #     pose = np.dot(previous_pose, pose)
        #
        #     view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, pose)
        #     vol_bnds_seq = np.zeros((3, 2))
        #     vol_bnds_seq[:, 0] = np.minimum(vol_bnds_seq[:, 0], np.amin(view_frust_pts, axis=1))
        #     vol_bnds_seq[:, 1] = np.maximum(vol_bnds_seq[:, 1], np.amax(view_frust_pts, axis=1))
        #     tsdf_vol_seq = fusion.TSDFVolume(vol_bnds_seq, voxel_size=voxel_size)
        #     tsdf_vol_seq.integrate(color_im, depth_im, cam_intr, pose, obs_weight=1.)
        #     second_Points3D = tsdf_vol_seq.get_point_cloud()[:, 0:3]
        #
        #     # 누적 pointcloud vertex only
        #     # first_Points3D = tsdf_vol.get_partial_point_cloud()
        #     first_Points3D = tsdf_vol.get_point_cloud()[:, 0:3]
        #
        #     pts_size = min(first_Points3D.shape[0], second_Points3D.shape[0])
        #     size = min(pts_size, 30000)
        #     samples_first = random.sample(range(first_Points3D.shape[0]), size)
        #     samples_second = random.sample(range(second_Points3D.shape[0]), size)
        #     pose_real, _, _ = icp(second_Points3D[samples_second, :],first_Points3D[samples_first, :])  # A, B // maps A onto B : B = pose*A
        #     pose_real = np.dot(previous_pose, pose_real)
        #
        #     cam_pose = pose_real
        #     previous_pose = cam_pose
        #     previous_Points3D = Points3D

            # output = joint_model(color_im)
            # joints, color_filtered, depth_filtered = filter_joint(output, color_im, depth_im)
            # for vol, color_, depth_ in zip(joints_vols, color_filtered, depth_filtered):
            #     vol.integrate(color_, depth_, cam_intr, cam_pose, obs_weight=1.)

            # joints_3D.append(joint_to_3D(joint, np.linalg.inv(cam_intr), cam_pose, depth_im))

        # poses.append(previous_pose)
        # Integrate observation into voxel volume (assume color aligned with depth)


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')  # Axe3D object
    joint_info = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                  'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

    # # Segmentation
    # for i in range(0, n_imgs):
    #     print("Human Body Vertex %d/%d" % (i + 1, n_imgs))
    #     depth_im = list_depth_im[i]
    #     color_im = list_color_im[i]
    #     output = model(color_im)
    #     # visualization
    #     # v = Visualizer(color_im[:, :, ::-1], MetadataCatalog.get(model.cfg.DATASETS.TRAIN[0]), scale=1.2)
    #     # v = v.draw_instance_predictions(output["instances"].to("cpu"))
    #     # cv2.imwrite(rf'0_sample_video\data_detectron\{i}_orig.jpg',color_im)
    #     # cv2.imwrite(rf'0_sample_video\data_detectron\{i}_seg.jpg',v.get_image()[:,:,::-1])
    #     not_valid_x, not_valid_y = filter_human(output)
    #     for not_x, not_y in zip(not_valid_x, not_valid_y):
    #         depth_im[not_x, not_y] = 0
    #         color_im[not_x, not_y] = 0
    #     print('Depth info')
    #     val_x, val_y = np.nonzero(depth_im)
    #
    #     threshold = np.mean(depth_im[val_x, val_y]) + 2 * np.std(depth_im[val_x, val_y])
    #     depth_im[depth_im >= threshold] = 0
    #     pose = poses[i]
    #     human_vol.integrate(color_im, depth_im, cam_intr, pose, obs_weight=1.)
    #
    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    # verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("human_mesh_seg_6.ply", verts, faces, norms, colors)

    verts_ = verts.T

    # for i in range(17):
    #     ax.scatter(mesh_joint[0, i], mesh_joint[1, i], mesh_joint[2, i], c='magenta') # projection  P = 4XN
    #     ax.text(mesh_joint[0, i], mesh_joint[1, i], mesh_joint[2, i], joint_info[i], fontsize=10)
    # for j in range(int(len(verts) / 3)):
    #     ax.scatter(verts_[0, 3* j], verts_[1, 3*j], verts_[2, 3*j], c='blue')
    # plt.show()

    # # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    # print("Saving point cloud")
    # point_cloud = tsdf_vol.get_point_cloud()
    # fusion.pcwrite("human_pcd_seg_4.ply", point_cloud)
