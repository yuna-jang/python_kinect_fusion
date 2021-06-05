import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
import numpy as np
import cv2
# try:
# from ICP import *
# except Exception as err:
#     from icp_modules.ICP import *

from helpers import colorize, convert_to_bgra_if_required
from pyk4a import PyK4APlayback
import pyk4a
from pyk4a import Config, PyK4A
import fusion



K_color = np.array([[614.76220703, 0, 637.63220215],
                    [0, 614.7354126, 369.26763916],
                    [0, 0, 1]])

K_depth = np.array([[504.76144409, 0, 323.20385742],
                    [0, 504.86602783, 330.52233887],
                    [0, 0, 1]])

K_color_inv = np.linalg.inv(K_color)
K_depth_inv = np.linalg.inv(K_depth)
D_W, D_H = 640, 576
C_W, C_H = 1920, 1080
R_T_I_0 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])


def play(playback: PyK4APlayback):
    count = 0
    poses = []
    img_list = []
    depth_list = []
    compare = []
    while count <= 50:
        try:
            capture = playback.get_next_capture()
            if count == 0:
                prev_d = None
                next_d = capture.depth_point_cloud.transpose(2, 0, 1).reshape(3, -1)
                compare.append(next_d)
            else:
                prev_d = next_d.copy()
                next_d = capture.depth_point_cloud.transpose(2, 0, 1).reshape(3, -1)
                pose, distances, _ = icp(prev_d.T, next_d.T)
                if len(poses) == 0:
                    poses.append(pose)
                else:
                    pose = poses[-1].dot(pose)
                    poses.append(pose)
                depth_list.append(capture.transformed_depth.astype(float))
                img_list.append(convert_to_bgra_if_required(playback.configuration["color_format"],
                                                            capture.color))
            if count % 5 == 0:
                print('Frame info. work ...... ', count)
            key = cv2.waitKey(10)
            if key != -1:
                break
        except EOFError:
            break
        count += 1
    # visualize(pose, prev_d, next_d)
    # poses = np.array(poses)
    # print(poses.shape)
    # fig = plt.figure(figsize=(8, 8))
    # ax = plt.axes(projection='3d')
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax.scatter(compare[0][0, :], compare[0][1, :], compare[0][2, :], c='g', s=0.3)
    # ax.scatter(compare[1][0, :], compare[1][1, :], compare[1][2, :], c='r', s=0.3)
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.scatter(compare[0][0, :], compare[0][1, :], compare[0][2, :], c='g', s=0.3)
    # ax.scatter(next_d[0, :], next_d[1, :], next_d[2, :], c='r', s=0.3)
    # plt.show()

    cv2.destroyAllWindows()
    return img_list, depth_list, poses


def filter_human(output):
    classes = output["instances"].pred_classes
    human = list(np.nonzero(np.where(classes.numpy() == 0, 1, 0))[0])
    boxes = output["instances"].pred_boxes
    focus = boxes.area().numpy()[human].argmax()
    mask = output["instances"].pred_masks[human[focus]]
    x, y = np.nonzero(1 - mask.numpy())
    return x, y


def play_tsdf(color_imgs:list, depth_imgs:list, poses: list, model):
    count = len(color_imgs)
    print('count', count)
    vol_bnds = np.zeros((3,2))
    for i in range(count):
        color_img = color_imgs[i]
        output = model(color_img)
        not_valid_x, not_valid_y = filter_human(output)
        depth_img = depth_imgs[i]
        for not_x, not_y in zip(not_valid_x, not_valid_y):
            depth_img[not_x, not_y] = 0
        depth_img = np.divide(depth_img, 1000)
        print('Get View frustum', '  ------------', i )
        depth_img[depth_img == 65.535] = 0
        depth_imgs[i] = depth_img
        pose = poses[i]
        view_frust_pts = fusion.get_view_frustum(depth_img, K_color, pose)
        print('-------------------------------')
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    print(vol_bnds)
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.1)

    for i in range(count):
        color_img = color_imgs[i]
        depth_img = depth_imgs[i]
        pose = poses[i]
        print('Integrate   ----------', i)
        tsdf_vol.integrate(color_img, depth_img, K_color, pose, obs_weight=1.)

    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite("pc.ply", point_cloud)


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


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


model = SEG_model()
filename = r'C:\Users\82106\PycharmProjects\dino_lib\python_kinect_fusion\tsdf-fusion-python-master\yuna2.mkv'
offset = 0

playback = PyK4APlayback(filename)
playback.open()

info(playback)

if offset != 0.0:
    playback.seek(int(offset * 1000000))

color_imgs, depth_imgs, poses = play(playback)
play_tsdf(color_imgs, depth_imgs, poses, model)
playback.close()
