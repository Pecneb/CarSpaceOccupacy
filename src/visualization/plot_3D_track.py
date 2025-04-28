import numpy as np
import matplotlib.pyplot as plt
import cv2
import fire
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_trajectory(trajectory_file, K_file):
    """
    Plot the 3D backprojected trajectory of the rear lights.

    Args:
        trajectory_file (str): Path to .npy file with tracked points (2, N, 2).
        K_file (str): Path to YAML file with K camera intrinsics.
    """
    # Load trajectories
    trajectories = np.load(trajectory_file)
    light1 = trajectories[0]  # shape (N, 2)
    light2 = trajectories[1]  # shape (N, 2)

    # Load camera intrinsics
    fs = cv2.FileStorage(K_file, cv2.FILE_STORAGE_READ)
    K = fs.getNode('K').mat()
    fs.release()
    K_inv = np.linalg.inv(K)

    # Homogenize and backproject points
    def backproject(point_2d):
        p_h = np.array([point_2d[0], point_2d[1], 1.0])
        ray = K_inv @ p_h
        ray /= np.linalg.norm(ray)  # Normalize for visualization
        return ray

    rays1 = np.array([backproject(pt) for pt in light1])
    rays2 = np.array([backproject(pt) for pt in light2])

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(rays1[:, 0], rays1[:, 1], rays1[:, 2], 'r-', label='Rear Light 1')
    ax.plot(rays2[:, 0], rays2[:, 1], rays2[:, 2], 'b-', label='Rear Light 2')

    ax.scatter(rays1[0, 0], rays1[0, 1], rays1[0, 2], c='r', marker='o', label='Start Light 1')
    ax.scatter(rays2[0, 0], rays2[0, 1], rays2[0, 2], c='b', marker='o', label='Start Light 2')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Backprojected Trajectories')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    fire.Fire(plot_3d_trajectory)