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


def filter_human(output):
    classes = output["instances"].pred_classes
    human = list(np.nonzero(np.where(classes.numpy() == 0, 1, 0))[0])
    boxes = output["instances"].pred_boxes
    if len(boxes.area().numpy()[human]) > 0:
        focus = boxes.area().numpy()[human].argmax()
        mask = output["instances"].pred_masks[human[focus]]
        x, y = np.nonzero(1 - mask.numpy())
        human = True
        return x, y, human
    else:
        human = False
        x = None
        y = None
        return x, y, human



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


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")
    return int(playback.length / 1000000)


if __name__ == "__main__":
    model = SEG_model()
    joint_model = Joint_model()
    # panoptic = panoptic_model()

    # Open kinect camera by realtime
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            color_format=pyk4a.ImageFormat.COLOR_MJPG
        )
    )

    # Load video file
    filename = r'C:\Users\82106\PycharmProjects\dino_lib\python_kinect_fusion\tsdf-fusion-python-master\yuna2.mkv'
    save_dir = r'C:\Users\82106\PycharmProjects\dino_lib\python_kinect_fusion\tsdf-fusion-python-master' \
               r'\segmentation_check'
    # n_frames = 5

    k4a = PyK4APlayback(filename)
    k4a.open()
    video_frames = info(k4a) * 30   # 30fps

    # Load Kinect's intrinsic parameter
    cam_intr = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)

    # List 생성
    list_depth_im = []
    list_color_im = []
    # vol_bnds 생성
    vol_bnds = np.zeros((3, 2))
    voxel_size = 0.01
    # while True:
    for i in range(video_frames):
        capture = k4a.get_next_capture()
        if capture.depth is not None and capture.color is not None: #and i%2==0:
            if i < 553:
                continue
            print(f"==========={i}==========")
            # Read depth and color image
            depth_im = capture.transformed_depth.astype(float)
            depth_im /= 1000.  ## depth is saved in 16-bit PNG in millimeters
            depth_im[depth_im >= 2] = 0  # set invalid depth to 0 (specific to 7-scenes dataset) 65.535=2^16/1000
            color_capture = convert_to_bgra_if_required(k4a.configuration["color_format"], capture.color)
            color_im = cv2.cvtColor(color_capture, cv2.COLOR_BGR2RGB)

            filtered_depth = cv2.bilateralFilter(depth_im.astype(np.float32), 5, 35, 35)
            filename = str(save_dir) + '\depth' + str(i) + '.jpg'
            cv2.imwrite(filename, filtered_depth)
            # color_im = cv2.bilateralFilter(color_im, 10, 50, 50)
            # sharpening_2 = np.array([[-1, -1, -1, -1, -1],
            #                          [-1, 2, 2, 2, -1],
            #                          [-1, 2, 9, 2, -1],
            #                          [-1, 2, 2, 2, -1],
            #                          [-1, -1, -1, -1, -1]]) / 9.0
            # color_im = cv2.filter2D(color_im, -1, sharpening_2)
            #
            # # Segmentaion human
            # output = model(color_im)
            # not_valid_x, not_valid_y, human = filter_human(output)
            # if not human:
            #     continue
            # for not_x, not_y in zip(not_valid_x, not_valid_y):
            #     depth_im[not_x, not_y] = 0
            #     color_im[not_x, not_y] = 0
            # # cv2.imshow('color', color_im)
            # # cv2.waitKey(0)
            # # list_depth_im.append(depth_im)
            # # list_color_im.append(color_im)
            # name = str(save_dir) + '\seg' + str(i) + '.jpg'
            # print(name)
            # cv2.imwrite(name, color_im)




