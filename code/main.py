import numpy as np
from utils import *
from scipy.linalg import expm


def vec2so3(vec):
    vec_hat = np.array([[0, -vec[2], vec[1]],
                        [vec[2], 0, -vec[0]],
                        [-vec[1], vec[0], 0]])
    return vec_hat


def vec2se3(vec):
    v = vec[:3]
    w = vec[3:]
    w_hat = vec2so3(w)
    vec_hat = np.r_[np.c_[w_hat, v],
                    np.zeros((1, 4))]
    return vec_hat


def vec2adj(vec):
    v = vec[:3]
    w = vec[3:]
    w_hat = vec2so3(w)
    v_hat = vec2so3(v)
    zero = np.zeros((3, 3))
    vec_adj = np.r_[np.c_[w_hat, v_hat],
                    np.c_[zero, w_hat]]
    return vec_adj


def trans_inv(mat):
    R = mat[:3, :3]
    R_T = np.transpose(R)
    p = mat[:3, 3]
    p = np.reshape(p, (3, 1))
    inv_mat = np.r_[np.c_[R_T, -np.matmul(R_T, p)], [[0, 0, 0, 1]]]
    return inv_mat


def projection_derivative(vec):
    mat = np.identity(4)
    mat[:, 2] = np.array([[-vec[0] / vec[2], -vec[1] / vec[2], 0, -vec[3] / vec[2]]]).T.reshape(4, )
    return mat


def dot_map(vec):
    s = vec[:3]
    s_hat = vec2so3(s)
    mat = np.r_[np.c_[np.identity(3), -s_hat], np.zeros((1, 6))]
    return mat


def imu_predict(mu, sig, tau, u):
    W = 0.0001 * np.identity(6)
    u_hat = vec2se3(u)
    u_adj = vec2adj(u)
    mu = np.matmul(expm(-tau * u_hat), mu)
    sig = np.matmul(expm(-tau * u_adj), sig)
    sig = np.matmul(sig, np.transpose(expm(-tau * u_adj))) + W
    return mu, sig


def map_update(imu_mu, map_mu, map_sig, features, K, b, o_T_i):
    m = features.shape[1]
    fsb = K[0][0] * b
    M = np.c_[np.r_[K[:2, :], K[:2, :]], np.array([[0, 0, fsb, 0]]).T]
    P = np.c_[np.identity(3), np.array([[0, 0, 0]]).T]
    V = 3000
    obs = np.array([i for i, j in enumerate(features[0, :] > -1) if j])
    idx = features > -1
    z = features[idx].reshape((4, -1))
    H = np.zeros((4 * obs.shape[0], 3 * m))

    for i in range(obs.shape[0]):
        if np.any(map_mu[:, obs[i]] == 0):
            d = features[0, obs[i]] - features[2, obs[i]]
            q = np.ones((4, 1))
            q[2] = fsb / d
            q[1] = (q[2] * features[1, obs[i]] - K[1][2]) / K[1][1]
            q[0] = (q[2] * features[0, obs[i]] - K[0][2]) / K[0][0]
            inv_mat = trans_inv(np.matmul(o_T_i, imu_mu))
            map_mu[:, obs[i]] = np.matmul(inv_mat, q).reshape(4, )
            continue

        temp = np.matmul(o_T_i, np.matmul(imu_mu, map_mu[:, obs[i]]))
        d_pi = projection_derivative(temp)
        H_ij = np.matmul(M, d_pi)
        H_ij = np.matmul(H_ij, np.matmul(o_T_i, np.matmul(imu_mu, P.T)))
        H[4 * i: 4 * i + 4, 3 * obs[i]:3 * obs[i] + 3] = H_ij

    temp1 = np.matmul(o_T_i, np.matmul(imu_mu, map_mu[idx].reshape((4, -1))))
    pi_temp1 = temp1 / temp1[2, :]
    z_hat = np.matmul(M, pi_temp1)

    inverse = np.linalg.inv(np.matmul(H, np.matmul(map_sig, H.T)) + V * np.identity(4 * obs.shape[0]))
    K_gain = np.matmul(map_sig, np.matmul(H.T, inverse))
    map_sig = np.matmul((np.identity(3 * m) - np.matmul(K_gain, H)), map_sig)
    map_mu = map_mu + np.matmul(P.T, np.matmul(K_gain, (z - z_hat).reshape(-1, 1)).reshape(3, -1))
    return map_mu, map_sig


def slam_update(imu_mu, imu_sig, map_mu, map_sig, features, K, b, o_T_i):
    m = features.shape[1]
    fsb = K[0][0] * b
    M = np.c_[np.r_[K[:2, :], K[:2, :]], np.array([[0, 0, fsb, 0]]).T]
    P = np.c_[np.identity(3), np.array([[0, 0, 0]]).T]
    V = 5000000000
    map_sig[-6:, -6:] = imu_sig

    obs = np.array([i for i, j in enumerate(features[0, :] > -1) if j])
    idx = features > -1
    z = features[idx].reshape((4, -1))
    H = np.zeros((4 * obs.shape[0], 3 * m + 6))

    for i in range(obs.shape[0]):
        if np.any(map_mu[:, obs[i]] == 0):
            d = features[0, obs[i]] - features[2, obs[i]]
            q = np.ones((4, 1))
            q[2] = fsb / d
            q[1] = (q[2] * features[1, obs[i]] - K[1][2]) / K[1][1]
            q[0] = (q[2] * features[0, obs[i]] - K[0][2]) / K[0][0]
            inv_mat = trans_inv(np.matmul(o_T_i, imu_mu))
            map_mu[:, obs[i]] = np.matmul(inv_mat, q).reshape(4, )
            continue

        temp = np.matmul(o_T_i, np.matmul(imu_mu, map_mu[:, obs[i]]))
        d_pi = projection_derivative(temp)
        H_ij = np.matmul(M, d_pi)
        H_ij = np.matmul(H_ij, np.matmul(o_T_i, np.matmul(imu_mu, P.T)))
        H[4 * i: 4 * i + 4, 3 * obs[i]:3 * obs[i] + 3] = H_ij

        temp1 = np.matmul(imu_mu, map_mu[:, obs[i]].T)
        temp1_dot = dot_map(temp1)
        new_Hij = np.matmul(M, d_pi)
        new_Hij = np.matmul(new_Hij, np.matmul(o_T_i, temp1_dot))
        H[4 * i: 4 * i + 4, -6:] = new_Hij

    temp2 = np.matmul(o_T_i, np.matmul(imu_mu, map_mu[idx].reshape((4, -1))))
    pi_temp2 = temp2 / temp2[2, :]
    z_hat = np.matmul(M, pi_temp2)

    inverse = np.linalg.pinv(np.matmul(H, np.matmul(map_sig, H.T)) + V * np.identity(4 * obs.shape[0]))
    K_gain = np.matmul(map_sig, np.matmul(H.T, inverse))
    map_sig = np.matmul((np.identity(H.shape[1]) - np.matmul(K_gain, H)), map_sig)
    K_gz = np.matmul(K_gain, (z - z_hat).reshape(-1, 1))
    temp3 = K_gz[:-6]
    map_mu = map_mu + np.matmul(P.T, temp3.reshape(3, -1))
    temp3_hat = vec2se3(K_gz[-6:].reshape(6))
    imu_mu = np.matmul(expm(temp3_hat), imu_mu)
    return imu_mu, map_mu, map_sig


if __name__ == '__main__':
    filename = "./data/0027.npz"
    savename = "./file_0027/"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)

    idx = [i for i in range(0, features.shape[1], 10)]
    features = features[:, idx, :]

    # Initialize
    imu_mu = np.identity(4)
    imu_sig = 0.01 * np.identity(6)

    m = features.shape[1]
    map_mu = np.zeros((4, m))
    # map_sig = np.identity(3 * m)
    map_sig = 300 * np.identity(3 * m + 6)

    traj = np.zeros((4, 4, t.shape[1]))

    for i in range(1, t.shape[1]):
        v = linear_velocity[:, i]
        w = rotational_velocity[:, i]
        u_vec = np.concatenate((v, w))
        tau = t[:, i] - t[:, i - 1]

        # (a) IMU Localization via EKF Prediction
        imu_mu, imu_sig = imu_predict(imu_mu, imu_sig, tau, u_vec)

        # traj[:, :, i] = trans_inv(imu_mu)

        # (b) Landmark Mapping via EKF Update
        # map_mu, map_sig = map_update(imu_mu, map_mu, map_sig, features[:, :, i], K, b, cam_T_imu)

        # (c) Visual-Inertial SLAM
        imu_mu, map_mu, map_sig = slam_update(imu_mu, imu_sig, map_mu, map_sig, features[:, :, i], K, b, cam_T_imu)

        traj[:, :, i] = trans_inv(imu_mu)

    # You can use the function below to visualize the robot pose over time
    visualize_trajectory_2d(traj, map_mu, path_name="file_0027", show_ori=True)
