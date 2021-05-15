from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import cv2
try:
    from ICP import *
except Exception as err:
    from icp_modules.ICP import *

from helpers import colorize, convert_to_bgra_if_required
from pyk4a import PyK4APlayback
import pyk4a
from pyk4a import Config, PyK4A
import fusion


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


# k4a = PyK4A(
#     Config(
#         color_resolution=pyk4a.ColorResolution.RES_720P,
#         depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
#     )
# )
# k4a.start()
#
# intrinsic_color = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
# intrinsic_depth = k4a.calibration.get_camera_matrix(pyk4a.calibration.CalibrationType.DEPTH)
# distortion = k4a.calibration.get_distortion_coefficients(pyk4a.calibration.CalibrationType.COLOR)
# cam_intr = intrinsic_color
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
    ### visualization 용도
    count = 0
    compare = []
    # while True:
    while count < 600:
        try:
            capture = playback.get_next_capture()
            # print(capture.depth[50, 50])
            # print(capture.transformed_depth.shape) # 1080 1920
            # print(colorize(capture.transformed_depth, (None, 5000)).shape) # 1080, 1920
            # if capture.color is not None:
            #     print(count, 'color ok')
            #     cv2.imshow("Color", capture.color)
                # cv2.imshow("Color", convert_to_bgra_if_required(playback.configuration["color_format"], capture.transformed_color))
            if capture.depth is not None:
                print(count, 'depth ok')

                PC = capture.depth_point_cloud.transpose(2, 0, 1).reshape(3, -1)
                if count == 0 or count == 300:
                    compare.append(PC)
                # print(PC.shape)
                # ax.scatter(PC[0, :], PC[1, :], PC[2, :])

                plt.show()
                # cv2.imshow("Depth", colorize(capture.depth, (None, 5000)))
                # cv2.imshow("TD", capture.transformed_depth)
            key = cv2.waitKey(10)
            if key != -1:
                break
        except EOFError:
            break
        count += 1

    plt.show()
    cv2.destroyAllWindows()


def play3(playback: PyK4APlayback):
    count = 0
    poses = []
    img_list = []
    depth_list = []
    compare = []
    while count <= 200:
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
                # print(pose)
                if len(poses) == 0:
                    # key_pose = pose
                    print(pose)
                    poses.append(pose)
                else:
                    # print('btw pose', pose)
                    pose = poses[-1].dot(pose)
                    # print('global pose', pose)
                    poses.append(pose)
                if count == 20:
                    print(pose)
                    print(np.vstack((next_d, np.ones((1, 262144)))).shape)
                    compare.append(pose.dot(np.vstack((next_d, np.ones((1, 262144))))))
                # depth_list.append(capture.transformed_depth.astype(float))
                # img_list.append(convert_to_bgra_if_required(playback.configuration["color_format"],
                #                                             capture.color))
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
    # # ax = plt.axes(projection='3d')
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax.scatter(compare[0][0, :], compare[0][1, :], compare[0][2, :], c='g', s=0.3)
    # ax.scatter(compare[1][0, :], compare[1][1, :], compare[1][2, :], c='r', s=0.3)
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.scatter(compare[0][0, :], compare[0][1, :], compare[0][2, :], c='g', s=0.3)
    # ax.scatter(next_d[0, :], next_d[1, :], next_d[2, :], c='r', s=0.3)
    # plt.show()
    # cv2.destroyAllWindows()
    return img_list, depth_list, poses


def DepthMAP(depth_im):
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0
    # depth_map = cv2.bilateralFilter(depth_im, 5, 35, 35).astype(np.float32)
    return depth_im


def play_tsdf(color_imgs:list, depth_imgs:list, poses: list):
    count = len(color_imgs)
    print('count', count)
    vol_bnds = np.zeros((3,2))
    for i in range(count):
        depth_img = depth_imgs[i]
        depth_img = np.divide(depth_img, 1000)
        print('-----------  ', i, '  ------------')
        print(depth_img.max(), depth_img.min())
        depth_img[depth_img == 65.535] = 0
        pose = poses[i]
        print('Pose')
        print(pose)
        view_frust_pts = fusion.get_view_frustum(depth_img, K_color, pose)
        print(view_frust_pts)
        print('-------------------------------')
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    print(vol_bnds)
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.05)

    for i in range(count):
        color_img = color_imgs[i]
        depth_img = depth_imgs[i]
        pose = poses[i]
        tsdf_vol.integrate(color_img, depth_img, K_color, pose, obs_weight=1.)

    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite("pc.ply", point_cloud)


def visualize(pose, point3D_prev, point3D_next):
    fig = plt.figure(figsize=(8, 8))
    # ax = plt.axes(projection='3d')
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    nums = int(point3D_next.shape[1] * 0.5)
    point_h = np.vstack((point3D_next, np.ones((1, point3D_next.shape[1]))))   # (x,y,z,1)
    point_h_trans = pose.dot(point_h)   # camera pose * (x,y,z,1)
    ax.scatter(point_h_trans[0, :nums], point_h_trans[1, :nums], point_h_trans[2, :nums], c='b', s=0.3)
    # ax.scatter(point3D_prev[0, :30000], point3D_prev[1, :30000], point3D_prev[2, :30000], c='b', s=0.3)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(point3D_next[0, :nums], point3D_next[1, :nums], point3D_next[2, :nums], c='r', s=0.3)
    plt.show()


def data_save(poses: list, color_imgs: list, depth_imgs: list, save_dir):
    for i in range(len(poses)):
        pose_name = save_dir + '\\frame_' + str(i).zfill(4) + '_pose.txt'
        color_name = save_dir + '\\frame_' + str(i).zfill(4) + '_color.jpg'
        depth_name = save_dir + '\\frame_' + str(i).zfill(4) + '_depth.png'
        pose = poses[i]
        color = color_imgs[i]
        depth = depth_imgs[i]
        np.savetxt(pose_name, pose)
        cv2.imwrite(color_name, color)
        cv2.imwrite(depth_name, depth)


def main() -> None:

    ### python playback.py --seek 0 영상경로/영상이름.mkv
    # parser = ArgumentParser(description="pyk4a player")
    # parser.add_argument("--seek", type=float, help="Seek file to specified offset in seconds", default=0.0)
    # parser.add_argument("FILE", type=str, help="Path to MKV file written by k4arecorder")
    #
    # args = parser.parse_args()
    # filename: str = args.FILE
    # offset: float = args.seek
    filename = 'C:\\Users\\82106\\PycharmProjects\\dino_lib\\azure\\sample2.mkv'
    filename = r'0_sample_video\sample2.mkv'

    offset = 0

    playback = PyK4APlayback(filename)
    playback.open()

    info(playback)

    # if offset != 0.0:
    #     playback.seek(int(offset * 1000000))
    play(playback)
    play3(playback)
    color_imgs, depth_imgs, poses = play3(playback)
    play_tsdf(color_imgs, depth_imgs, poses)
    data_save(poses, color_imgs=color_imgs, depth_imgs=depth_imgs,
           save_dir='')
    # playback.close()

if __name__ == "__main__":
    main()
