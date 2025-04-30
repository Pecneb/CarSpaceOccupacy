import numpy as np
import fire
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D


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

    # Homogenize points
    def to_homogeneous(p):
        return np.array([p[0], p[1], 1.0])

    L1h = to_homogeneous(L1)
    R1h = to_homogeneous(R1)
    L2h = to_homogeneous(L2)
    R2h = to_homogeneous(R2)

    print(f"Homogeneous coordinates:")
    print(f"L1h: {L1h}")
    print(f"R1h: {R1h}")
    print(f"L2h: {L2h}")
    print(f"R2h: {R2h}")

    # Compute lines of vanishing point Vx
    Vx_h = np.cross(L1h, R1h)
    Vx_k = np.cross(L2h, R2h)

    # Compute lines of vanishing point Vy
    Vy_h = np.cross(L1h, L2h)
    Vy_k = np.cross(R1h, R2h)

    # Compute vanishing points
    Vx = np.cross(Vx_h, Vx_k)
    Vy = np.cross(Vy_h, Vy_k)

    # Normalize vanishing points
    Vx /= Vx[2]
    Vy /= Vy[2]

    # Print vanishing points
    print(f"Vanishing Point Vy (intersection of L1-L2): {Vy}")
    print(f"Vanishing Point Vx (intersection of R1-R2): {Vx}")

    # Calculate vanishing line l_inf

    l_inf = np.cross(Vx, Vy)

    # Backproject into 3D directions using K inverse
    K_inv = np.linalg.inv(K)
    Vy_3D = K_inv @ Vy
    Vx_3D = K_inv @ Vx

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
        l_inf_3D = K_inv @ l_inf
        l_inf_3D /= np.linalg.norm(l_inf_3D)  # Normalize for visualization

        # Plot l_inf_3D
        ax_3d.quiver(
            0, 0, 0, l_inf_3D[0], l_inf_3D[1], l_inf_3D[2], color="g", label="l_inf_3D", linewidth=2
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
