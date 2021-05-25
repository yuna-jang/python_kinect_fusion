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
    s = sum(b.T.dot(a) for a, b in zip(AA, BB)) / sum(a.T.dot(a) for a in AA)
    # translation
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

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)

    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()



def icp(A, B, init_pose=None, max_iterations=15, tolerance=0.001):
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
    # print(A.shape, B.shape) # 대략 30만개 X 3
    assert A.shape == B.shape
    # select Nonzero(=outlier)
    Ao = A.shape[0]
    Bo = B.shape[0]
    # nonA = np.nonzero(np.sum(A, axis=1))
    # nonB = np.nonzero(np.sum(B, axis=1))
    # A = A[nonA, :]  # 3xn
    # B = B[nonB, :]  # 3xn
    # A = np.squeeze(A)
    # B = np.squeeze(B)
    # get number of dimensions
    # print('=======ICP=======')
    # print(A.shape)
    # print(B.shape)
    # print('=================')
    m = A.shape[1]
    N = min(A.shape[0], B.shape[0])
    size = min(A.shape[0], B.shape[0])
    sampler = random.sample(range(N), size)

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1, size))
    dst = np.ones((m+1, size))
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

        # compute the transformation between the current source and nearest destination points
        # T, _, _ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)
        T, _, _, s = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = T.dot(src) * s
        # # update the current source
        # src = np.dot(T, src)
        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    # print('calculate final transformation')
    # calculate final transformation
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
def ICP_point_to_plane():
    pass


def FindRigidTransform(points_set_P, points_set_Q):
    """
    :param points_set_P: 3D points cloud, ndarray 3*N
    :param points_set_Q: 3D points cloud, ndarray 3*N
    :return: Trans [R|t] from P to Q, shape=(3,4)
    """
    P = np.mean(points_set_P, axis=1, keepdims=True)
    Q = np.mean(points_set_Q, axis=1, keepdims=True)
    # mass center transform
    X = points_set_P - P
    Y = points_set_Q - Q
    print(X.shape)
    print(Y.shape)
    M = X.dot(Y.T)
    # print(M.shape)
    # M = X.dot(Y)
    # print(M)
    U, Sigma, Vt = np.linalg.svd(M)
    R = np.dot(Vt.T, U.T)
    t = Q - R.dot(P)
    trans = np.hstack((R, t))
    # print('Trans', trans)
    return trans
    pass


def FindMatchingPairs(points_set_P,points_set_Q,pose, thresh=20):
    """
    :param points_set_P: 3D points cloud, ndarray 3*N
    :param points_set_Q: 3D points cloud, ndarray 3*N
    :param pose: Trans [R|t] from P to Q shape=(3,4)
    :param thresh: distance threshold of filtering matching pairs
    :return: matching pairs index -> ind_P,ind_Q
    """
    P = np.vstack((points_set_P, np.ones((1, points_set_P.shape[1]))))
    print()
    P_projection = pose.dot(P)
    kdt = KDTree(points_set_Q, metric='euclidean')
    dist, ind = kdt.query(P_projection, k=1, return_distance=True)
    # print('dist')
    # print(dist)
    ind_P = []
    ind_Q = []
    mean_error = 0
    for i in range(len(dist)):
        print(dist[i])
        # if dist[i] < thresh:
        ind_P.append(i)
        ind_Q.append(ind[i][0])
        mean_error += dist[i]
    # mean_error /= len(ind_P)
    print('ind_P', ind_P)
    print('ind_Q', ind_Q)
    return ind_P, ind_Q
    pass


def ICP_point_to_point(points_set_P, points_set_Q):
    """
    :param points_set_P: 3D points cloud, ndarray 3*N
    :param points_set_Q: 3D points cloud, ndarray 3*N
    :return: Trans [R|t] from P to Q, shape is (4,4),the last row is [0|1]
    iteration times = 20, find the best trans matrix between neighboring frames
    """
    iter_times = 20
    pose = FindRigidTransform(points_set_P, points_set_Q)
    ind_P, ind_Q = FindMatchingPairs(points_set_P, points_set_Q, pose)
    matching_num = len(ind_P)
    # print('First matching num',matching_num)
    for i in range(iter_times):
        temp_P = points_set_P[:,ind_P]
        temp_Q = points_set_Q[:,ind_Q]
        temp_pose = FindRigidTransform(temp_P, temp_Q)
        temp_ind_P, temp_ind_Q = FindMatchingPairs(points_set_P, points_set_Q, pose)
        temp_matching_num = len(temp_ind_P)
        if temp_matching_num > matching_num:
            pose = temp_pose
            ind_P = temp_ind_P
            ind_Q = temp_ind_Q
            matching_num = temp_matching_num
        else:
            break
    # print('pose', pose)
    return np.vstack((pose, np.array([0,0,0,1])))
    pass