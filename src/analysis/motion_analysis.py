from typing import Tuple, Union, Dict, Any, Optional
import numpy as np
import fire
import matplotlib.pyplot as plt
import cv2
import yaml
from mpl_toolkits.mplot3d import Axes3D


class MotionAnalysis:
    def __init__(
        self,
        selected_points_file: str,
        K_file: str,
        car_params_file: Optional[str] = None,
    ):
        """
        Initialize the MotionAnalysis class with common arguments.

        Args:
            selected_points_file (str): Path to the selected points (L1, R1, L2, R2).
            K_file (str, optional): Path to camera intrinsic matrix (YAML file saved by OpenCV FileStorage). If None, assumes identity.
        """
        self.selected_points_file = selected_points_file
        self.K_file = K_file
        self.K = self._load_camera_matrix()
        self.car_params_file = car_params_file
        self.car_params = self._load_car_params()

    def _load_camera_matrix(self) -> np.ndarray:
        """Load the camera intrinsic matrix from the K_file or use the identity matrix."""
        if self.K_file:
            fs = cv2.FileStorage(self.K_file, cv2.FILE_STORAGE_READ)
            K = fs.getNode("K").mat()
            fs.release()
        else:
            K = np.eye(3)
        return K

    def _load_car_params(self) -> Dict[str, Any]:
        """Load YAML file containing car parameters"""
        if self.car_params_file:
            with open(self.car_params_file) as f:
                car_params_dict = yaml.safe_load(f)
                return car_params_dict
        else:
            return {
                "brand": "Audi",
                "model": "A3 Sportback",
                "version": "5-door",
                "dimensions": {
                    "length_mm": 4450,
                    "width_mm": 1816,
                    "height_mm": 1430,
                },
                "wheelbase_mm": 2630,
                "track_width_mm": {
                    "front": 1560,
                    "rear": 1540,
                },
                "ground_clearance_mm": 140,
            }

    def to_homogeneous(self, p: np.ndarray) -> np.ndarray:
        """Convert Euclidean coordinate to Homogeneous coordinate."""
        return np.array([p[0], p[1], 1.0])

    def compute_vanishing_points(
        self,
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
        Compute the vanishing points Vx and Vy.

        Args:
            l1 (np.ndarray): Homogeneous coordinates of the first left point.
            r1 (np.ndarray): Homogeneous coordinates of the first right point.
            l2 (np.ndarray): Homogeneous coordinates of the second left point.
            r2 (np.ndarray): Homogeneous coordinates of the second right point.
            intermediate_results (bool): If True, return intermediate results.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
                If intermediate_results is False, returns (Vx, Vy).
                If intermediate_results is True, returns (Vx, Vy, Vx_h, Vx_k, Vy_h, Vy_k).
        """
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

    def _plot_results_2D(
        self,
        L1: np.ndarray,
        R1: np.ndarray,
        L2: np.ndarray,
        R2: np.ndarray,
        Vx: np.ndarray,
        Vy: np.ndarray,
        Vx_h: np.ndarray,
        Vx_k: np.ndarray,
        Vy_h: np.ndarray,
        Vy_k: np.ndarray,
        l_inf: np.ndarray,
    ) -> None:
        """
        Helper function to plot 2D results for vanishing points analysis.

        This function visualizes the relationships between points and lines in a 2D space,
        including vanishing points and their corresponding lines, using matplotlib.

        Args:
            L1 (np.ndarray): Homogeneous Coordinates of the left point in the first frame (shape: (3,)).
            R1 (np.ndarray): Homogeneous Coordinates of the right point in the first frame (shape: (3,)).
            L2 (np.ndarray): Homogeneous Coordinates of the left point in the second frame (shape: (3,)).
            R2 (np.ndarray): Homogeneous Coordinates of the right point in the second frame (shape: (3,)).
            Vx (np.ndarray): Homogeneous Coordinates of the vanishing point in the x-direction (shape: (3,)).
            Vy (np.ndarray): Homogeneous Coordinates of the vanishing point in the y-direction (shape: (3,)).
            Vx_h (np.ndarray): Homogeneous representation of the line passing through L1 and R1 (shape: (3,)).
            Vx_k (np.ndarray): Homogeneous representation of the line passing through L2 and R2 (shape: (3,)).
            Vy_h (np.ndarray): Homogeneous representation of the line passing through L1 and L2 (shape: (3,)).
            Vy_k (np.ndarray): Homogeneous representation of the line passing through R1 and R2 (shape: (3,)).
            l_inf (np.ndarray): Homogeneous representation of the line at infinity (shape: (3,)).

        Returns:
            None: This function does not return any value. It displays a 2D plot.
        """
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

    def _plot_results_3D(
        self, Vx: np.ndarray, Vy: np.ndarray, l_inf: np.ndarray
    ) -> None:
        """
        Helper function to plot 3D vectors and visualize their relationships.

        This function creates a 3D plot to visualize the input vectors `Vx`, `Vy`,
        and `l_inf` in a 3D space. It normalizes `l_inf` for better visualization
        and sets appropriate plot limits to ensure all vectors are displayed clearly.

        Args:
            Vx (np.ndarray): A 3-element numpy array representing the first 3D vector.
            Vy (np.ndarray): A 3-element numpy array representing the second 3D vector.
            l_inf (np.ndarray): A 3-element numpy array representing the third 3D vector
                to be normalized and visualized.

        Returns:
            None: This function does not return any value. It displays a 3D plot.
        """
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
        ax_3d.quiver(
            0,
            0,
            0,
            l_inf_norm[0],
            l_inf_norm[1],
            l_inf_norm[2],
            color="g",
            label="l_inf_3D",
        )
        ax_3d.set_title("3D Visualization")
        ax_3d.legend()
        plt.show()

    def estimate_camera_distance_cosine(
        self, ray1: np.ndarray, ray2: np.ndarray, distance: float
    ) -> float:
        """
        Theorem of Cosines Representation:

                 Camera
                    O
                   / \
                  /   \
                 /     \
                /       \
               /         \
              /           \
             /             \
            /_______________\
           Left Rear Light   Right Rear Light
            A                  B

        Triangle:
        - O is the camera lens (apex of the triangle).
        - A and B are the left and right rear lights, respectively.
        - The angle θ is the angle between the two viewing rays (OA and OB).
        - The distance AB is the side opposite to the angle θ.

        Theorem of Cosines:
        AB² = OA² + OB² - 2 * OA * OB * cos(θ)

        Args:
            ray1 (np.ndarray): Left Rear Light in 3D space [X, Y, Z, 1]
            ray2 (np.ndarray): Right Rear Light in 3D space [X, Y, Z, 1]
            distance (float): The distance of the two rear lights in mm.

        Returns:
            float: The scale of the which the rays have to be multiplied by.
        """
        ray1_norm = ray1 / np.linalg.norm(ray1)
        ray2_norm = ray2 / np.linalg.norm(ray2)
        cos_theta = np.dot(ray1_norm, ray2_norm)
        return distance / np.sqrt(2 * (1 - cos_theta))

    def camera_to_plane_distance(self, plot_results: bool = True) -> None:
        """Computer and plot distance from camera to plane PI which is the plane that the car rear lights are on.

        Args:
            plot_results (bool, optional): Plot results?. Defaults to True.
        """
        # Initialize left and right rear light coordinates
        coordinates = np.load(self.selected_points_file)
        L1, R1, _, _ = coordinates

        # Make the coorindates homogeneous
        L1 = np.array([L1[0], L1[1], 1])
        R1 = np.array([R1[0], R1[1], 1])

        # Load car parameters and estimate rear light distance (6 cm overhand on both sides) (scale it to M instead of MM)
        light_distance = (self.car_params["track_width_mm"]["rear"] + 2 * 60) / 1e3

        # Backproject into 3D space
        L1_3D, R1_3D = self.backproject_point_to_3D_ray(np.stack([L1, R1]))

        # Normalize
        L1_3D /= np.linalg.norm(L1_3D)
        R1_3D /= np.linalg.norm(R1_3D)

        # Estimate scale
        scale = self.estimate_camera_distance_cosine(L1_3D, R1_3D, light_distance)
        print(f"The scale of the triangle: {scale}")

        if plot_results:
            self._plot_cosine_triangle(L1_3D, R1_3D, scale)

    def _plot_cosine_triangle(
        self, ray1: np.ndarray, ray2: np.ndarray, scale: float
    ) -> None:
        """
        Plot the triangle formed by the camera and two scaled rays using the law of cosines.

        Args:
            ray1: np.array, direction vector to left light (unit)
            ray2: np.array, direction vector to right light (unit)
            scale: float, computed distance from camera to each light
        """
        # Scale the rays
        P1 = ray1 * scale
        P2 = ray2 * scale
        origin = np.array([0, 0, 0])

        # Prepare plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot camera to points
        ax.plot(
            [origin[0], P1[0]],
            [origin[1], P1[1]],
            [origin[2], P1[2]],
            "b-",
            label="r_L",
        )
        ax.plot(
            [origin[0], P2[0]],
            [origin[1], P2[1]],
            [origin[2], P2[2]],
            "g-",
            label="r_R",
        )

        # Connect the two points (real-world light distance)
        ax.plot(
            [P1[0], P2[0]],
            [P1[1], P2[1]],
            [P1[2], P2[2]],
            "r--",
            label="d (real distance)",
        )

        # Style
        ax.scatter(*origin, color="black", s=50, label="Camera")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Triangle from Camera to Rear Lights (Law of Cosines)")
        ax.legend()
        ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    fire.Fire(MotionAnalysis)
