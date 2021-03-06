import numpy as np
import cv2
import time
import random


def DepthMap(deepth):
    # 双边滤波
    # filtered = cv2.bilateralFilter(deepth.astype(np.float32), 5, 15, 15)
    filtered = cv2.bilateralFilter(deepth.astype(np.float32), 5, 35, 35)
    pass
    return filtered


def PointCloud(depth, Inverse):
    """
    :param depth: DepthMap of one frame
    :param camIn: Camera Intrinsic inverse K^{-1}
    :return: 3D Points Set shape->3*N
    """
    depth = DepthMap(depth) # Bilateral Filter
    h, w = depth.shape
    num = h*w
    v_ind, u_ind = np.nonzero(depth)
    samples = random.sample(range(len(v_ind)), 10000)
    v_samples = v_ind[samples]
    u_samples = u_ind[samples]
    dmap = depth.reshape((1, num))

    depth_ind = []
    for i in range(len(v_samples)):
        v = v_samples[i]
        u = u_samples[i]
        index = v*w+u
        depth_ind.append(index)
    dmap = dmap[:, depth_ind]
    uvdMap = np.vstack((u_samples, v_samples, dmap))
    Points = np.dot(Inverse, uvdMap)
    return Points
    pass


def NormalMap(Points,h,w):
    # 越界的法向量被设置成了(0,0,0),可能需要处理
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
    return np.array(normal_map)


if __name__ == '__main__':
    Points = np.zeros((3, 800))
    norm = NormalMap(Points, 20, 40)