import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib.pyplot as plt

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
    M = X.dot(Y.T)

    U, Sigma, Vt = np.linalg.svd(M)
    R = np.dot(Vt.T, U.T)
    t = Q - R.dot(P)
    trans = np.hstack((R, t))
    return trans
    pass


def FindMatchingPairs(points_set_P,points_set_Q,pose, thresh=0.01):
    """
    :param points_set_P: 3D points cloud, ndarray 3*N
    :param points_set_Q: 3D points cloud, ndarray 3*N
    :param pose: Trans [R|t] from P to Q shape=(3,4)
    :param thresh: distance threshold of filtering matching pairs
    :return: matching pairs index -> ind_P,ind_Q
    """
    P = np.vstack((points_set_P, np.ones((1, points_set_P.shape[1]))))
    P_projection = pose.dot(P)
    kdt = KDTree(points_set_Q.T, metric='euclidean')
    dist, ind = kdt.query(P_projection.T, k=1, return_distance=True)
    # print('dist')
    # print(dist)
    ind_P = []
    ind_Q = []

    mean_error = 0
    for i in range(len(dist)):
        if dist[i] < thresh:
            ind_P.append(i)
            ind_Q.append(ind[i][0])
            mean_error += dist[i]
    return ind_P, ind_Q
    pass


def nearest_neighbor(points_set_P,points_set_Q,pose, thresh=0.005):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    P = np.vstack((points_set_P, np.ones((1, points_set_P.shape[1]))))
    P_projection = pose.dot(P)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(points_set_Q.T) #dst
    dist, ind = neigh.kneighbors(P_projection.T, return_distance=True) #src
    ind_P = []
    ind_Q = []
    for i in range(len(dist)):
        if dist[i] < thresh:
            ind_Q.append(i)
            ind_P.append(ind[i][0])
    return ind_P, ind_Q
    pass
    # return dist.ravel(), ind.ravel()



def ICP_point_to_point(points_set_P, points_set_Q):
    """
    :param points_set_P: 3D points cloud, ndarray 3*N
    :param points_set_Q: 3D points cloud, ndarray 3*N
    :return: Trans [R|t] from P to Q, shape is (4,4),the last row is [0|1]
    iteration times = 20, find the best trans matrix between neighboring frames
    """
    iter_times = 100
    pose = FindRigidTransform(points_set_P, points_set_Q)
    # ind_P, ind_Q = FindMatchingPairs(points_set_P, points_set_Q, pose)
    ind_P, ind_Q = nearest_neighbor(points_set_P, points_set_Q, pose)
    matching_num = len(ind_P)
    print('First matching num',matching_num)
    for i in range(iter_times):
        temp_P = points_set_P[:,ind_P]
        temp_Q = points_set_Q[:,ind_Q]
        temp_pose = FindRigidTransform(temp_P, temp_Q)
        # temp_ind_P, temp_ind_Q = FindMatchingPairs(points_set_P, points_set_Q, pose)
        temp_ind_P, temp_ind_Q = nearest_neighbor(points_set_P, points_set_Q, pose)
        temp_matching_num = len(temp_ind_P)
        if temp_matching_num >= matching_num:
            pose = temp_pose
            ind_P = temp_ind_P
            ind_Q = temp_ind_Q
            matching_num = temp_matching_num
        else:
            break
    # print('pose',pose)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 매칭포인트만 띄우기
    # temp_P = np.vstack((temp_P, np.ones((1, temp_P.shape[1]))))
    # temp_P_trans = pose.dot(temp_P)
    # ax.scatter(temp_Q[0, :], temp_Q[1, :], temp_Q[2, :], c='g', s=0.3)
    # # ax.scatter(temp_P[0, :], temp_P[1, :], temp_P[2, :], c='b', s=0.3)
    # ax.scatter(temp_P_trans[0, :], temp_P_trans[1, :], temp_P_trans[2, :], c='r', s=0.3)

    # 전체포인트 띄우기
    points_set_P = np.vstack((points_set_P, np.ones((1, points_set_P.shape[1]))))
    temp_P_trans = pose.dot(points_set_P)
    ax.scatter(points_set_Q[0, :], points_set_Q[1, :], points_set_Q[2, :], c='g', s=0.3)
    # ax.scatter(points_set_P[0, :], points_set_P[1, :], points_set_P[2, :], c='b', s=0.3)
    ax.scatter(temp_P_trans[0, :], temp_P_trans[1, :], temp_P_trans[2, :], c='r', s=0.3)
    plt.show()

    return np.vstack((pose, np.array([0,0,0,1])))
    pass

if __name__ == '__main__':
    pass