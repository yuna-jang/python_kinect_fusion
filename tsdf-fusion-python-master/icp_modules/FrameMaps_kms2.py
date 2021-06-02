import numpy as np
import cv2
import time
import random


def DepthMap(deepth):
    ## deepth : depth image. W*H. 각 픽셀 값에는 RGB 데이터대신 depth 값
    filtered = cv2.bilateralFilter(deepth.astype(np.float32), 5, 15, 15)
    pass
    return filtered


def PointCloud(depth, Inverse):
    """
    :param depth: DepthMap of one frame
    :param camIn: Camera Intrinsic inverse K^{-1} : 3x3
    :return: 3D Points Set shape : (WH) * 3
    """

    depth = DepthMap(depth)  # Bilateral Filter. W*H
    H, W = depth.shape

    v_ind, u_ind = np.nonzero(depth)  # 행 index / 열 index
    nonzero_idx = np.array([v_ind, u_ind]).T
    sampler = random.sample(range(len(nonzero_idx)), min(200000, len(nonzero_idx)))
    nonzero_sampler = nonzero_idx[sampler, :]
    point_homo = np.ones((3, 200000))
    for i in range(len(nonzero_sampler)):
        yy, xx = nonzero_sampler[i]
        d = depth[yy, xx]
        point_homo[:2, i] = xx * d, yy * d
    cloud = np.dot(Inverse, point_homo).T  # 3X3 @ 3XN -> 3XN  ==> T ==> NX3
    return cloud, nonzero_sampler


def NormalMap(sampler, depth_img, Inverse):
    """
    :param sampler: Nonzero indexes made from PointCloud()
    :param depth_img: 2D depth image
    :param Inverse: inverse of intrinsic camera matrix
    :return: Normal Map
    v[(V(u+1, v) - V(u, v)) X (V(u, v+1) - V(u,v))]
    """
    start_time = time.time()
    h, w = depth_img.shape
    normal_map = np.zeros((3, len(sampler)))
    for i in range(len(sampler)):
        yy, xx = sampler[i]
        vec1 = depth_img[yy, xx] * np.dot(Inverse, np.array([xx, yy, 1]).T)
        if yy + 1 >= h:
            vec2 = np.array([0.0, 0.0, 0.0])
        else:
            vec2 = depth_img[yy + 1, xx] * np.dot(Inverse, np.array([xx, yy + 1, 1]).T)
        if xx + 1 >= w:
            vec3 = np.array([0.0, 0.0, 0.0])
        else:
            vec3 = depth_img[yy, xx + 1] * np.dot(Inverse, np.array([xx + 1, yy, 1]).T)
        normal = np.cross(vec2 - vec1, vec3 - vec1)
        abs_val = np.sum(np.square(normal))
        if abs_val != 0:
            normal_map[:, i] = normal / abs_val
        else:
            normal_map[:, i] = 0

    print('Time Used in NormalMap', time.time() - start_time)
    # normal_map shape = w * h * 3
    return normal_map.T


if __name__ == '__main__':
    Points = np.zeros((3, 640 * 480))
    norm = NormalMap(Points, 20, 40)
