import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
import cv2
import random
import matplotlib.pyplot as plt

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
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    # print('BestFitTransform')
    return T, R, t


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

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    # print('NearestNeighbors')
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=30, tolerance=0.001):
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
    m = A.shape[1]
    N = min(A.shape[0], B.shape[0])
    size = min(3000, A.shape[0], B.shape[0])
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
        # print('ICP iteration: ', i)
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)
        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    # print('calculate final transformation')
    # calculate final transformation
    T, _, _ = best_fit_transform(A[sampler, :], src[:m, :].T)

    return T, distances, i

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 매칭포인트만 띄우기
    # # temp_P = np.vstack((temp_P, np.ones((1, temp_P.shape[1]))))
    # # temp_P_trans = pose.dot(temp_P)
    # # ax.scatter(temp_Q[0, :], temp_Q[1, :], temp_Q[2, :], c='g', s=0.3)
    # # # ax.scatter(temp_P[0, :], temp_P[1, :], temp_P[2, :], c='b', s=0.3)
    # # ax.scatter(temp_P_trans[0, :], temp_P_trans[1, :], temp_P_trans[2, :], c='r', s=0.3)
    #
    # # 전체포인트 띄우기
    # points_set_P = np.vstack((points_set_P, np.ones((1, points_set_P.shape[1]))))
    # temp_P_trans = pose.dot(points_set_P)
    # ax.scatter(points_set_Q[0, :], points_set_Q[1, :], points_set_Q[2, :], c='g', s=0.3)
    # # ax.scatter(points_set_P[0, :], points_set_P[1, :], points_set_P[2, :], c='b', s=0.3)
    # ax.scatter(temp_P_trans[0, :], temp_P_trans[1, :], temp_P_trans[2, :], c='r', s=0.3)
    # plt.show()

if __name__ == '__main__':
    pass