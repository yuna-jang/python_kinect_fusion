import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from icp_modules.FramePreprocessing import *
from scipy.linalg import svd


def nearest_neighbor(src, dst):
    # kNN : KDTree 로 correspondence points 를 찾음.
    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1, algorithm="kd_tree",
                             metric="euclidean")
    neigh.fit(dst)

    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def A_matrix(src, normal):
    A = np.zeros((src.shape[0], 6))
    for i in range(len(A)):
        A[i, 0] = normal[i, 2]*src[i, 1] - normal[i, 1]*src[i, 2]
        A[i, 1] = normal[i, 0]*src[i, 2] - normal[i, 2]*src[i, 0]
        A[i, 2] = normal[i, 1]*src[i, 0] - normal[i, 0]*src[i, 1]
        A[i, 3] = normal[i, 0]
        A[i, 4] = normal[i, 1]
        A[i, 5] = normal[i, 2]
    return A


def B_vector(src, dst, normal):
    B = np.zeros((src.shape[0], 1))
    for i in range(len(B)):
        B[i] = np.sum(np.multiply(normal[i, :], dst[i, :])) \
                - np.sum(np.multiply(normal[i, :], src[i, :]))
    return B


def inv_S(S, r, c):
    assert r == c
    arr = np.zeros((r, c))
    for i in range(len(S)):
        if S[i] != 0:
            arr[i, i] = 1 / S[i]

    return arr


def M_matrix(x):
    M = np.eye(4)
    M[0, 1] = -x[2]
    M[0, 2] = x[1]
    M[0, 3] = x[3]
    M[1, 0] = x[2]
    M[1, 2] = -x[0]
    M[1, 3] = x[4]
    M[2, 0] = -x[1]
    M[2, 1] = x[0]
    M[2, 3] = x[5]
    return M    # 4X4


def point_to_plane(src, dst, normal, iteration=20):
    """
    src: source 3D point cloud :: Nx3
    dst: destination 3D point cloud :: Nx3
    normal: destination's normal map :: Nx3
    """
    # N = min(src.shape[0], dst.shape[0])
    # size = min(src.shape[0], dst.shape[0], 50000)
    print(src.shape, dst.shape, normal.shape)
    batch = 20

    T_history = []
    error_history = []

    for b in range(batch):
        dist, idx = nearest_neighbor(src, dst)
        dist = list(dist)
        idx = list(idx)
        mem = {}
        for d, i in zip(dist, idx):
            mem[d] = i
        sampler = []
        dist_sort = sorted(dist)
        for i in range(20000):
            sampler.append(mem[dist_sort[i]])
        src_batch = np.copy(src[sampler, :])
        dst_batch = np.copy(dst[sampler, :])
        normal_batch = np.copy(normal[sampler, :])
        T = np.eye(4)
        for iter_ in range(iteration):
            A = A_matrix(src_batch, normal_batch)
            B = B_vector(src_batch, dst_batch, normal_batch)
            # U, S, Vt = np.linalg.svd(A, full_matrices=False)
            U, S, Vt = svd(A, full_matrices=False)
            invS = inv_S(S, U.shape[1], Vt.shape[0])
            invA = Vt.T @ invS @ U.T
            x_opt = np.dot(invA, B)
            error = np.sum(np.square(np.dot(A, x_opt) - B)) / src.shape[0]

            M = M_matrix(x_opt)     # 4x4
            T = np.dot(T, M)

            src_b_homo = np.vstack((src_batch.T, np.ones((1, src_batch.shape[0]))))     # 4XN
            src_T = np.dot(M, src_b_homo)   # 4XN
            src_batch = src_T[:-1, :].T
        error_history.append(error)
        T_history.append(T)
        src_homo = np.vstack((src.T, np.ones((1, src.shape[0]))))
        src_T = np.dot(T, src_homo)
        src = src_T[:-1, :].T
        print("Batch {} ::".format(b), error)

    best = T_history[int(np.argmin(error_history))]
    return best
