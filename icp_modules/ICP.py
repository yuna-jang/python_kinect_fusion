import numpy as np
from sklearn.neighbors import KDTree
import cv2
import random
from sklearn.neighbors import NearestNeighbors

###
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    # print(A.shape, B.shape)
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
    # translation
    # t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    # print('BestFitTransform')
    return T, R, t, s


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape
    # np.where(np.isnan(src))
    # src = np.nan_to_num(src)
    # dst = np.nan_to_num(dst)

    # neigh = NearestNeighbors(n_neighbors=1)
    neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
    neigh.fit(dst)

    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()



def icp(A, B, init_pose=None, max_iterations=50, tolerance=1e-7):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        -> Nx3 array (N??? point ??????)
        B: Nxm numpy array of destination mD point
        -> Nx3 array (N??? point ??????)
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''
    # print(A.shape, B.shape)
    assert A.shape == B.shape
    m = A.shape[1]
    N = min(A.shape[0], B.shape[0])
    size = min(A.shape[0], B.shape[0])
    # o_size = size
    # batch = 20
    # batch_tolerance = 1e-9
    # batch_error = []
    # make points homogeneous, copy them to maintain the originals
    # for b in range(batch):
    src = np.ones((m+1, size))
    dst = np.ones((m+1, size))
    sampler = random.sample(range(N), size)
    src[:m, :] = np.copy(A[sampler, :].T)
    dst[:m, :] = np.copy(B[sampler, :].T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        src = np.nan_to_num(src)
        dst = np.nan_to_num(dst)
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m,:].T)
        src = np.nan_to_num(src)
        dst = np.nan_to_num(dst)

        T, _, _, s = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = T.dot(src) * s
        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            print('Error', mean_error)
            break
        prev_error = mean_error
    A = np.nan_to_num(A)
    src = np.nan_to_num(src)
    T, _, _, s = best_fit_transform(A[sampler, :], src[:m, :].T)
    return T, distances, i


# def icp_select(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
#     assert A.shape == B.shape
#     # get number of dimensions
#     sample = 3000
#
#     # make points homogeneous, copy them to maintain the originals
#     # for i in range(max_iterations):
#     #     for j in range(batch_loop):
#
#     src = np.ones((m + 1, A.shape[0]))
#     dst = np.ones((m + 1, B.shape[0]))
#     src[:m, :] = np.copy(A.T)
#     dst[:m, :] = np.copy(B.T)
#
#     # apply the initial pose estimation
#     if init_pose is not None:
#         src = np.dot(init_pose, src)
#
#     prev_error = 0
#
#     for i in range(max_iterations):
#         print('ICP iteration: ', i)
#         # find the nearest neighbors between the current source and destination points
#         distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)
#
#         # compute the transformation between the current source and nearest destination points
#         T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)
#
#         # update the current source
#         src = np.dot(T, src)
#
#         # check error
#         mean_error = np.mean(distances)
#         if np.abs(prev_error - mean_error) < tolerance:
#             break
#         prev_error = mean_error
#
#     # calculate final transformation
#     T, _, _ = best_fit_transform(A, src[:m, :].T)
#
#     return T, distances, i


###