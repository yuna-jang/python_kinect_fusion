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

    depth = DepthMap(depth)     # Bilateral Filter. W*H
    H, W = depth.shape
    point_homo = np.ones((3, H*W))
    v_ind, u_ind = np.nonzero(depth)
    nonzero_idx = np.array([v_ind, u_ind]).T
    col = 0
    for h in range(H):
        if h % 100 == 0:
            print("Generating Point Cloud...", h)
        for w in range(W):
            if (h, w) in nonzero_idx:
                point_homo[2, col] = 0
                d = 1
            else:
                d = depth[h, w]
            point_homo[:2, col] = h * d, w * d
            col += 1
    cloud = np.dot(Inverse, point_homo).T       # 3X3 @ 3XN -> 3XN  ==> T ==> NX3
    return cloud


def NormalMap(Points, h=480, w=640):
    ### Points : 3XN
    print(Points.shape)
    start_time = time.time()
    normal_map = []
    for i in range(h):
        normal_map.append([])
        for j in range(w):
            index = i*w+j
            vec1 = Points[:,index]
            if i+1 >= h:
                vec2 = np.array([0.0,0.0,0.0])
            else:
                index2 = (i+1)*w+j
                vec2 = Points[:, index2]
            if j+1 >= w:
                vec3 = np.array([0.0,0.0,0.0])
            else:
                index3 = i*w+j+1
                vec3 = Points[:, index3]
            normal = np.cross(vec2-vec1, vec3-vec1)
            normal_map[i].append(normal)
    print('Time Used in NormalMap', time.time()-start_time)
    # normal_map shape = w * h * 3
    W, H, _ = np.array(normal_map).shape
    return np.array(normal_map).reshape(W*H, 3)


if __name__ == '__main__':
    Points = np.zeros((3, 640*480))
    norm = NormalMap(Points, 20, 40)
