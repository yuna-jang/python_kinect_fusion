import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from icp_modules.FramePreprocessing import *

def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    # print('Vt', Vt.shape)
    R = np.dot(Vt.T, U.T)
    # print('R', R.shape)
    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)
    # translation
    s = sum(b.T.dot(a) for a, b in zip(AA, BB)) / sum(a.T.dot(a) for a in AA)
    t = centroid_B.T - s * np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t, s


def nearest_neighbor(src, dst):
    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1, algorithm="kd_tree",
                             metric="euclidean")
    neigh.fit(dst)

    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()



def icp(A, B, init_pose=None, max_iterations=30, tolerance=0.000001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        -> Nx3 array (N은 point 개수)
        B: Nxm numpy array of destination mD point
        -> Nx3 array (N은 point 개수)
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''
    assert A.shape == B.shape
    m = A.shape[1]
    N = min(A.shape[0], B.shape[0])
    size = min(A.shape[0], B.shape[0], 4000)
    batch = 10
    for b in range(batch):
        sampler = random.sample(range(N), size)

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m+1, size))
        dst = np.ones((m+1, size))
        src[:m, :] = np.copy(A[sampler, :].T)
        dst[:m, :] = np.copy(B[sampler, :].T)
        prev_error = 0
        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            src = np.nan_to_num(src)
            dst = np.nan_to_num(dst)
            distances, indices = nearest_neighbor(src[:m, :].T, dst[:m,:].T)

            T, _, _, s = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

            src = T.dot(src) * s
            mean_error = np.mean(distances)

            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error
        print('Batch', b, 'ME', prev_error)
    A = np.nan_to_num(A)
    src = np.nan_to_num(src)
    T, _, _, s = best_fit_transform(A[sampler, :], src[:m, :].T)

    return T, distances, i


def point_plane_energy(dmap, prev_vert, next_vert,
                       prev_norm, prev_pose, th_d, th_o):
    energy = 0
    H = 480
    W = 640
    # null 판단
    idxs = []
    for h in range(H):
        for w in range(W):
            depth = dmap[h, w]
            if depth != 0:  # Omega(u) != Null 를 만족하는 u 넣기
                idxs.append(W*h + w)





