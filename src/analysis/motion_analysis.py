from typing import Tuple, Union
import numpy as np
import fire
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D


def compute_vanising_points_Vx_and_Vy(
    l1: np.ndarray,
    r1: np.ndarray,
    l2: np.ndarray,
    r2: np.ndarray,
    intermediate_results: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Compute the vanishing points Vx and Vy from the given homogeneous coordinates
    of the left and right rear lights captured in two video frames.
    The vanishing points are calculated based on the intersection of lines formed
    by the given points in homogeneous coordinates.
    Args:
        l1 (np.ndarray): Homogeneous coordinates of the left rear light in frame 1.
        r1 (np.ndarray): Homogeneous coordinates of the right rear light in frame 1.
        l2 (np.ndarray): Homogeneous coordinates of the left rear light in frame 2.
        r2 (np.ndarray): Homogeneous coordinates of the right rear light in frame 2.
        intermediate_results (bool): Set to True if want the function to return intermediate results h, k rays.
    Returns:
        Tuple of Vx, Vy as numpy arrays
        or
        Tuple of Vx, Vy, Vx_h, Vx_k, Vy_h, Vy_k if intermediate_results argument is set to True
    """
    # Compute lines of vanishing point Vx
    Vx_h = np.cross(l1, r1)
    Vx_k = np.cross(l2, r2)

    # Compute lines of vanishing point Vy
    Vy_h = np.cross(l1, l2)
    Vy_k = np.cross(r1, r2)

    # Compute vanishing points
    Vx = np.cross(Vx_h, Vx_k)
    Vy = np.cross(Vy_h, Vy_k)
    if intermediate_results:
        return Vx, Vy, Vx_h, Vx_k, Vy_h, Vy_k
    return Vx, Vy


def to_homogeneous(p: np.ndarray) -> np.ndarray:
    """Convert Euclidean coordinate to Homogeneous coordinate

    Args:
        p (np.ndarray): Euclidean coordinate

    Returns:
        np.ndarray: Homogeneous coordinate
    """
    return np.array([p[0], p[1], 1.0])


def computer_vanishing_line(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute the vanishing line from vanishing points p and q.

    Args:
        p (np.ndarray): Vanishing point 1
        q (np.ndarray): Vanishing points 2

    Returns:
        np.ndarray: Vanishing line
    """
    return np.cross(p, q)


def backproject_coordinate(x: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Backproject coordinate p with intrinsic matrix K

    Args:
        p (np.ndarray): Homogeneous coordinate
        K (np.ndarray): Intrinsic parameters matrix
        batch (bool): Batch toggle

    Returns:
        np.ndarray: Backprojected coordinate or coordinates if batch toggle is True
    """
    K_inv = np.linalg.inv(K)
    if len(x.shape) == 2:
        Y = np.empty_like(x)
        for i in range(x.shape[0]):
            Y[i] = np.dot(K_inv, x[i])
        return Y
    else:
        return np.dot(K_inv, x)


def analyze_motion(selected_points_file, K_file=None, plot_result=True):
    """
    Analyze motion type (translation or steering) from selected rear light points.
    Args:
        selected_points_file (str): Path to the selected points (L1, R1, L2, R2).
        K_file (str, optional): Path to camera intrinsic matrix (YAML file saved by OpenCV FileStorage). If None, assumes identity.
        plot_result (bool): If True, show a plot of the geometric situation.
    """
    # Load selected points
    points = np.load(selected_points_file)  # shape (4, 2)
    if points.shape != (4, 2):
        raise ValueError("Selected points must have shape (4, 2)")

    L1, R1, L2, R2 = points

    # Optionally load camera matrix K
    if K_file:
        fs = cv2.FileStorage(K_file, cv2.FILE_STORAGE_READ)
        K = fs.getNode("K").mat()
        fs.release()
    else:
        K = np.eye(3)

    L1h = to_homogeneous(L1)
    R1h = to_homogeneous(R1)
    L2h = to_homogeneous(L2)
    R2h = to_homogeneous(R2)

    print(f"Homogeneous coordinates:")
    print(f"L1h: {L1h}")
    print(f"R1h: {R1h}")
    print(f"L2h: {L2h}")
    print(f"R2h: {R2h}")

    Vx, Vy, Vx_h, Vx_k, Vy_h, Vy_k = compute_vanising_points_Vx_and_Vy(
        L1h, R1h, L2h, R2h, True
    )

    # Normalize vanishing points
    Vx /= Vx[2]
    Vy /= Vy[2]

    # Print vanishing points
    print(f"Vanishing Point Vy (intersection of L1-L2): {Vy}")
    print(f"Vanishing Point Vx (intersection of R1-R2): {Vx}")

    # Compute vanishing line
    l_inf = computer_vanishing_line(Vx, Vy)

    # Backproject vanishing coordinates and line in 3D
    Vy_3D, Vx_3D, l_inf_3D = backproject_coordinate(np.stack([Vy, Vx, l_inf]), K)

    print(f"3D Coordinates of Vy: {Vy_3D}")
    print(f"3D Coordinates of Vx: {Vx_3D}")

    # Check orthogonality
    dot_product = np.dot(Vx_3D[:2], Vy_3D[:2])
    angle_cos = dot_product / (np.linalg.norm(Vx_3D[:2]) * np.linalg.norm(Vy_3D[:2]))

    print(f"Dot product of Vx and Vy directions: {dot_product:.4f}")

    print(f"Cosine of angle between Vx and Vy directions: {angle_cos:.4f}")

    if abs(angle_cos) < 0.1:
        print("\nResult: The car is TRANSLATING FORWARD.")
    else:
        print("\nResult: The car is STEERING (turning).")

    if plot_result:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([L1[0], R1[0]], [L1[1], R1[1]], "g-o", label="Frame 1")
        ax.plot([L2[0], R2[0]], [L2[1], R2[1]], "m-o", label="Frame 2")
        ax.plot(Vx[0], Vx[1], "rx", markersize=10, label="Vx")
        ax.plot(Vy[0], Vy[1], "bx", markersize=10, label="Vy")
        x_vals = np.linspace(
            np.min([Vx[0], Vy[0]]), np.max([Vx[0], Vy[0]])
        )  # Take min and max values of vanishing points, so we can see both on the plot.
        y_vals_Vx_h = -(Vx_h[0] * x_vals + Vx_h[2]) / Vx_h[1]
        y_vals_Vx_k = -(Vx_k[0] * x_vals + Vx_k[2]) / Vx_k[1]
        y_vals_Vy_h = -(Vy_h[0] * x_vals + Vy_h[2]) / Vy_h[1]
        y_vals_Vy_k = -(Vy_k[0] * x_vals + Vy_k[2]) / Vy_k[1]
        ax.plot(x_vals, y_vals_Vx_h, "g--", label="Vx_h (L1-R1)")
        ax.plot(x_vals, y_vals_Vx_k, "m--", label="Vx_k (L2-R2)")
        ax.plot(x_vals, y_vals_Vy_h, "b--", label="Vy_h (L1-L2)")
        ax.plot(x_vals, y_vals_Vy_k, "b--", label="Vy_k (R1-R2)")
        y_vals_l_inf = -(l_inf[0] * x_vals + l_inf[2]) / l_inf[1]
        ax.plot(x_vals, y_vals_l_inf, "r-", label="l_inf")
        ax.set_title("Vanishing Points Analysis")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.legend()
        plt.grid()
        plt.show()

        # 3D Plot for Vx_3D, Vy_3D, and backprojected l_inf
        fig_3d = plt.figure(figsize=(8, 8))
        ax_3d = fig_3d.add_subplot(111, projection="3d")

        # Plot Vx_3D and Vy_3D
        ax_3d.quiver(
            0, 0, 0, Vx_3D[0], Vx_3D[1], Vx_3D[2], color="r", label="Vx_3D", linewidth=2
        )
        ax_3d.quiver(
            0, 0, 0, Vy_3D[0], Vy_3D[1], Vy_3D[2], color="b", label="Vy_3D", linewidth=2
        )

        # Backproject l_inf into 3D
        l_inf_3D /= np.linalg.norm(l_inf_3D)  # Normalize for visualization

        # Plot l_inf_3D
        ax_3d.quiver(
            0,
            0,
            0,
            l_inf_3D[0],
            l_inf_3D[1],
            l_inf_3D[2],
            color="g",
            label="l_inf_3D",
            linewidth=2,
        )

        # Set plot limits for better visualization
        max_range = max(
            np.abs(Vx_3D).max(), np.abs(Vy_3D).max(), np.abs(l_inf_3D).max()
        )
        ax_3d.set_xlim([-max_range, max_range])
        ax_3d.set_ylim([-max_range, max_range])
        ax_3d.set_zlim([-max_range, max_range])

        # Add labels and legend
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("3D Visualization of Vx_3D, Vy_3D, and l_inf_3D")
        ax_3d.legend()

        # Show the plot
        plt.show()


if __name__ == "__main__":
    fire.Fire(analyze_motion)
