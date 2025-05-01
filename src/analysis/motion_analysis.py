from typing import Tuple, Union
import numpy as np
import fire
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D


class MotionAnalysis:
    def __init__(self, selected_points_file: str, K_file: str):
        """
        Initialize the MotionAnalysis class with common arguments.

        Args:
            selected_points_file (str): Path to the selected points (L1, R1, L2, R2).
            K_file (str, optional): Path to camera intrinsic matrix (YAML file saved by OpenCV FileStorage). If None, assumes identity.
        """
        self.selected_points_file = selected_points_file
        self.K_file = K_file
        self.K = self._load_camera_matrix()

    def _load_camera_matrix(self) -> np.ndarray:
        """Load the camera intrinsic matrix from the K_file or use the identity matrix."""
        if self.K_file:
            fs = cv2.FileStorage(self.K_file, cv2.FILE_STORAGE_READ)
            K = fs.getNode("K").mat()
            fs.release()
        else:
            K = np.eye(3)
        return K

    def to_homogeneous(self, p: np.ndarray) -> np.ndarray:
        """Convert Euclidean coordinate to Homogeneous coordinate."""
        return np.array([p[0], p[1], 1.0])

    def compute_vanishing_points(self, l1, r1, l2, r2, intermediate_results=False):
        """Compute the vanishing points Vx and Vy."""
        Vx_h = np.cross(l1, r1)
        Vx_k = np.cross(l2, r2)
        Vy_h = np.cross(l1, l2)
        Vy_k = np.cross(r1, r2)
        Vx = np.cross(Vx_h, Vx_k)
        Vy = np.cross(Vy_h, Vy_k)
        if intermediate_results:
            return Vx, Vy, Vx_h, Vx_k, Vy_h, Vy_k
        return Vx, Vy

    def backproject_point_to_3D_ray(self, v_2d: np.ndarray) -> np.ndarray:
        """
        Backproject a 2D image point (e.g. vanishing point) to a 3D ray using K^{-1}.

        Args:
            v_2d (array): Homogeneous image point [x, y, 1]
            K (array): 3x3 camera intrinsic matrix

        Returns:
            ray_3d (array): 3D direction vector from camera center (not scaled)
        """
        K_inv = np.linalg.inv(self.K)
        if len(v_2d.shape) == 2:
            return np.dot(K_inv, v_2d.T).T
        return np.dot(K_inv, v_2d)

    def backproject_line_to_plane_normal(self, l_2d: np.ndarray) -> np.ndarray:
        """
        Backproject a 2D image line (e.g. vanishing line) to a 3D plane normal using K^T.

        Args:
            l_2d (array): Homogeneous image line [a, b, c]
            K (array): 3x3 camera intrinsic matrix

        Returns:
            n_3d (array): 3D plane normal vector (not necessarily unit norm)
        """
        l_h = np.array(l_2d)
        n = self.K.T @ l_h
        return n / np.linalg.norm(n)

    def analyze_motion(self, plot_result: bool = True):
        """
        Analyze motion type (translation or steering) from selected rear light points.
        """
        # Load selected points
        points = np.load(self.selected_points_file)  # shape (4, 2)
        if points.shape != (4, 2):
            raise ValueError("Selected points must have shape (4, 2)")

        L1, R1, L2, R2 = points
        L1h, R1h, L2h, R2h = map(self.to_homogeneous, [L1, R1, L2, R2])

        # Compute vanishing points
        Vx, Vy, Vx_h, Vx_k, Vy_h, Vy_k = self.compute_vanishing_points(
            L1h, R1h, L2h, R2h, intermediate_results=True
        )
        Vx /= Vx[2]
        Vy /= Vy[2]

        # Compute vanishing line
        l_inf = np.cross(Vx, Vy)

        print(f"K matrix: {self.K}")

        # Backproject vanishing points and line into 3D
        Vy_3D, Vx_3D = self.backproject_point_to_3D_ray(np.stack([Vy, Vx]))
        l_inf_3D = self.backproject_line_to_plane_normal(l_inf)

        # Analyze motion type
        dot_product = np.dot(Vx_3D[:2], Vy_3D[:2])
        angle_cos = dot_product / (
            np.linalg.norm(Vx_3D[:2]) * np.linalg.norm(Vy_3D[:2])
        )
        print(f"Dot product of Vx_3D[{Vx_3D}] @ Vy_3D[{Vy_3D}]: {dot_product}")
        print(f"Angle of two vectors: {np.degrees(np.cosh(np.abs(angle_cos)))}")
        if abs(angle_cos) < 0.1:
            print("\nResult: The car is TRANSLATING FORWARD.")
        else:
            print("\nResult: The car is STEERING (turning).")

        # Plot results if requested
        if plot_result:
            self._plot_results_2D(L1, R1, L2, R2, Vx, Vy, Vx_h, Vx_k, Vy_h, Vy_k, l_inf)
            self._plot_results_3D(Vx_3D, Vy_3D, l_inf_3D)

    def _plot_results_2D(self, L1, R1, L2, R2, Vx, Vy, Vx_h, Vx_k, Vy_h, Vy_k, l_inf):
        """Helper function to plot 2D."""
        # 2D Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot([L1[0], R1[0]], [L1[1], R1[1]], "g-o", label="Frame 1")
        ax.plot([L2[0], R2[0]], [L2[1], R2[1]], "m-o", label="Frame 2")
        ax.plot(Vx[0], Vx[1], "rx", markersize=10, label="Vx")
        ax.plot(Vy[0], Vy[1], "bx", markersize=10, label="Vy")

        x_vals = np.linspace(np.min([Vx[0], Vy[0]]), np.max([Vx[0], Vy[0]]))

        y_vals_l_inf = -(l_inf[0] * x_vals + l_inf[2]) / l_inf[1]
        y_vals_Vx_h = -(Vx_h[0] * x_vals + Vx_h[2]) / Vx_h[1]
        y_vals_Vx_k = -(Vx_k[0] * x_vals + Vx_k[2]) / Vx_k[1]
        y_vals_Vy_h = -(Vy_h[0] * x_vals + Vy_h[2]) / Vy_h[1]
        y_vals_Vy_k = -(Vy_k[0] * x_vals + Vy_k[2]) / Vy_k[1]

        ax.plot(x_vals, y_vals_Vx_h, "g--", label="Vx_h (L1-R1)")
        ax.plot(x_vals, y_vals_Vx_k, "m--", label="Vx_k (L2-R2)")
        ax.plot(x_vals, y_vals_Vy_h, "b--", label="Vy_h (L1-L2)")
        ax.plot(x_vals, y_vals_Vy_k, "b--", label="Vy_k (R1-R2)")
        ax.plot(x_vals, y_vals_l_inf, "r-", label="l_inf")

        ax.set_title("Vanishing Points Analysis")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.legend()

        plt.grid()
        plt.show()

    def _plot_results_3D(self, Vx, Vy, l_inf):
        """Helper function to plot 3D."""
        # 3D Plot
        fig_3d = plt.figure(figsize=(8, 8))
        ax_3d = fig_3d.add_subplot(111, projection="3d")

        # Plot Vx_3D and Vy_3D
        # ax_3d.quiver(
        #     0, 0, 0, Vx_3D[0], Vx_3D[1], Vx_3D[2], color="r", label="Vx_3D", linewidth=2
        # )
        # ax_3d.quiver(
        #     0, 0, 0, Vy_3D[0], Vy_3D[1], Vy_3D[2], color="b", label="Vy_3D", linewidth=2
        # )

        # Backproject l_inf into 3D
        l_inf_norm = l_inf / np.linalg.norm(l_inf)  # Normalize for visualization

        # Set plot limits for better visualization
        max_range = max(np.abs(Vx).max(), np.abs(Vy).max(), np.abs(l_inf_norm).max())
        ax_3d.set_xlim([-max_range, max_range])
        ax_3d.set_ylim([-max_range, max_range])
        ax_3d.set_zlim([-max_range, max_range])

        # Add labels and legend
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("3D Visualization of Vx_3D, Vy_3D, and l_inf_3D")

        # Plot 3D features
        ax_3d.quiver(0, 0, 0, Vx[0], Vx[1], Vx[2], color="r", label="Vx_3D")
        ax_3d.quiver(0, 0, 0, Vy[0], Vy[1], Vy[2], color="b", label="Vy_3D")
        ax_3d.quiver(0, 0, 0, l_inf_norm[0], l_inf_norm[1], l_inf_norm[2], color="g", label="l_inf_3D")
        ax_3d.set_title("3D Visualization")
        ax_3d.legend()
        plt.show()


if __name__ == "__main__":
    fire.Fire(MotionAnalysis)
