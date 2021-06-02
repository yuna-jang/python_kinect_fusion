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
from icp_modules.ICP_kms import *
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

def filter_joint(output):
    joints = output["instances"].pred_keypoints[0][:, :2].numpy()
    return joints


def Joint_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor


def joint_to_3D(joints, Inverse, pose, depth_im):
    Joints = np.zeros((3, 17))
    for i in range(17):
        xx, yy = joints[i]
        d = depth_im[int(round(yy)), int(round(xx))]
        U = d * np.dot(Inverse, np.array([xx, yy, 1]).T)
        Joints[:, i] = np.dot(pose, np.array([U[0], U[1], U[2], 1]))[:-1]
    return Joints


def simple_bundle(joint_3D: list):
    # Outlier 날리고 평균 구하기.
    joints_3D = np.array(joint_3D)  # N * 3 * 17
    if len(joint_3D) > 50:
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

    else:
        result = np.mean(joints_3D, axis=0)
    return result


if __name__ == "__main__":
    model = SEG_model()
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
    # filename = r'C:\Users\82106\PycharmProjects\dino_lib\python_kinect_fusion\video1.mkv'
    filename = r'C:\Users\82106\PycharmProjects\dino_lib\python_kinect_fusion\tsdf-fusion-python-master\human6.mkv'
    n_frames = 30


    k4a = PyK4APlayback(filename)
    k4a.open()

    # Load Kinect's intrinsic parameter
    cam_intr = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
    invK = np.linalg.inv(cam_intr)
    # List 생성
    list_depth_im = []
    list_color_im = []
    # vol_bnds 생성
    vol_bnds = np.zeros((3, 2))
    voxel_size = 0.03
    iter = 0
    # while True:
    for i in range(0, n_frames):
        capture = k4a.get_next_capture()
        if capture.depth is not None and capture.color is not None:
            print(f"==========={iter}==========")
            # Read depth and color image
            depth_im = capture.transformed_depth.astype(float)
            depth_im /= 1000.  ## depth is saved in 16-bit PNG in millimeters
            depth_im[depth_im > 5.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset) 65.535=2^16/1000
            color_capture = convert_to_bgra_if_required(k4a.configuration["color_format"], capture.color)
            color_im = cv2.cvtColor(color_capture, cv2.COLOR_BGR2RGB)

            list_depth_im.append(depth_im)
            list_color_im.append(color_im)

            if iter == 0:
                first_Points3D = PointCloud(depth_im, invK)
                cam_pose = np.eye(4)
                first_pose = cam_pose

            elif iter >= 1:
                second_Points3D = PointCloud(depth_im, invK)
                ind = random.sample(range(first_Points3D.shape[1]), second_Points3D.shape[1])
                pose, distances, _ = icp(second_Points3D.T,
                                         first_Points3D.T[ind, :])  # A, B // maps A onto B : B = pose*A
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
    human_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)
    k4a.close()
    poses = []
    # ===============Integrate===============
    n_imgs = len(list_depth_im)
    iter = 0
    joints_3D = []
    for iter in range(0, n_imgs):
        print("Fusing frame %d/%d" % (iter + 1, n_imgs))

        # Read depth and color image
        depth_im = list_depth_im[iter]
        color_im = list_color_im[iter]

        output = model(color_im)
        not_valid_x, not_valid_y = filter_human(output)
        for not_x, not_y in zip(not_valid_x, not_valid_y):
            depth_im[not_x, not_y] = 0
            color_im[not_x, not_y] = 0
        val_x, val_y = np.nonzero(depth_im)
        threshold = np.mean(depth_im[val_x, val_y]) + 2 * np.std(depth_im[val_x, val_y])
        depth_im[depth_im >= threshold] = 0


        # Set first frame as world system
        if iter == 0:
            previous_Points3D = PointCloud(depth_im, invK)
            cam_pose = np.eye(4)
            previous_pose = cam_pose

        elif iter == 1:
            second_Points3D = PointCloud(depth_im, invK)
            pose, distances, _ = icp(second_Points3D.T, previous_Points3D.T)  # A, B // maps A onto B : B = pose*A
            cam_pose = np.dot(previous_pose, pose)
            previous_pose = cam_pose
            previous_Points3D = second_Points3D

        elif iter > 1:
            Points3D = PointCloud(depth_im, invK)

            # Compute camera view frustum and extend convex hull
            pose, distances, _ = icp(Points3D.T, previous_Points3D.T)  # A, B // maps A onto B : B = pose*A
            pose = np.dot(previous_pose, pose)
            view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, pose)
            vol_bnds_seq = np.zeros((3, 2))
            vol_bnds_seq[:, 0] = np.minimum(vol_bnds_seq[:, 0], np.amin(view_frust_pts, axis=1))
            vol_bnds_seq[:, 1] = np.maximum(vol_bnds_seq[:, 1], np.amax(view_frust_pts, axis=1))
            tsdf_vol_seq = fusion.TSDFVolume(vol_bnds_seq, voxel_size=voxel_size)
            tsdf_vol_seq.integrate(color_im, depth_im, cam_intr, pose, obs_weight=1.)
            second_Points3D = tsdf_vol_seq.get_point_cloud()[:, 0:3]

            # 누적 pointcloud vertex only
            first_Points3D = tsdf_vol.get_partial_point_cloud()

            pts_size = min(first_Points3D.shape[0], second_Points3D.shape[0])
            pose, distances, _ = icp(second_Points3D[0:pts_size, :],
                                     first_Points3D[0:pts_size, :])  # A, B // maps A onto B : B = pose*A
            print(f'{pts_size} / {first_Points3D.shape[0]}')
            pose = np.dot(previous_pose, pose)

            cam_pose = pose
            previous_pose = cam_pose
            previous_Points3D = Points3D

            output = joint_model(color_im)
            joint = filter_joint(output)
            joints_3D.append(joint_to_3D(joint, invK, pose, depth_im))

        poses.append(previous_pose)
        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_im, depth_im, cam_intr, cam_pose, obs_weight=1.)
        iter = iter + 1

    # # Segmentation
    # for i in range(0, n_imgs):
    #     print("Human Body Vertex %d/%d" % (i + 1, n_imgs))
    #     depth_im = list_depth_im[i]
    #     color_im = list_color_im[i]
    #
    #     pose = poses[i]
    #     human_vol.integrate(color_im, depth_im, cam_intr, pose, obs_weight=1.)
    joint_ = simple_bundle(joints_3D)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')  # Axe3D object
    print(joint_.shape)
    print(joint_)

    joint_info = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                  'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

    for i in range(17):
        ax.scatter(joint_[0, i], joint_[1, i], joint_[2, i]) # projection  P = 4XN
        ax.text(joint_[0, i], joint_[1, i], joint_[2, i], joint_info[i], fontsize=10)
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
