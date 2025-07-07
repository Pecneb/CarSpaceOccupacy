import logging
import random
from logging import getLogger, Logger
from typing import Tuple, Union, Dict, Any, Optional, List
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
        tracked_features: str = "rear_lights",
        debug: bool = False,
    ):
        """Initializes the motion analysis object with the required configuration files and parameters.

        Args:
            selected_points_file (str): Path to the file containing selected points for analysis.
            K_file (str): Path to the file containing the camera intrinsic matrix.
            car_params_file (Optional[str], optional): Path to the file containing car parameters. Defaults to None.
            tracked_features (str, optional): The type of features to track ["rear_lights", "license_plate"]. Defaults to "rear_lights".
        """
        allowed_tracked_features = ["rear_lights", "license_plate"]
        if tracked_features not in allowed_tracked_features:
            raise ValueError(
                f"tracked_features must be one of {allowed_tracked_features}"
            )
        self._logger = self._init_logger(debug)
        self.selected_points_file = selected_points_file
        self.K_file = K_file
        self.K, self.dist_coeffs = self._load_camera_matrix()
        self.car_params_file = car_params_file
        self.car_params = self._load_car_params()
        self.tracked_features = tracked_features

    def _init_logger(self, debug: bool) -> Logger:
        logger = getLogger("MotionAnalysisLogger")
        if debug:
            logger.setLevel(logging.DEBUG)
            if not logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        else:
            logger.setLevel(logging.INFO)
            if not logger.hasHandlers():
                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        return logger

    def _load_camera_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the camera intrinsic matrix from the K_file or use the identity matrix."""
        # if self.K_file:
        #     fs = cv2.FileStorage(self.K_file, cv2.FILE_STORAGE_READ)
        #     K = fs.getNode("K").mat()
        #     fs.release()
        # else:
        #     K = np.eye(3)
        # Get calibration parameters
        # Load from file
        data = np.load(str(self.K_file))

        # Extract camera matrix and distortion coefficients
        K = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]

        self._logger.debug(
            "Read and Load calibration matrices: K %s and dist_coeffs %s",
            str(K),
            str(dist_coeffs),
        )
        return K, dist_coeffs

    def _load_car_params(self) -> Dict[str, Any]:
        """Load YAML file containing car parameters"""
        self._logger.debug("Loading car parameters %s", self.car_params_file)
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
        if len(p.shape) == 2 and p.shape[1] == 2:
            return np.hstack((p, np.ones(shape=(p.shape[0], 1))))
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

    def compute_vanishing_points_vectorized(
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute vanishing points from input tracked point pairs.

        Args:
            points (np.ndarray): numpy array of left and right tracked feature segments.

        Returns:
            Tuple[np.ndarray, np.ndarray]: a tuple containing the vanishing points.
        """
        vx_lines = np.array()

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

    def ransac_vanishing_point(
        self, lines: List[np.ndarray], threshold: float = 0.5, iterations: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate vanishing point from noisy line segments using RANSAC.

        Args:
            lines (List[np.ndarray]): List of 2D homogeneous line vectors [a, b, c]
            threshold (float): Distance threshold for inliers (in pixels)
            iterations (int): Number of RANSAC iterations

        Returns:
            best_vp (np.ndarray): Homogeneous coordinates of vanishing point
            best_inliers (List[int]): Indices of inlier lines
        """
        best_vp = None
        best_inliers = []

        for _ in range(iterations):
            # Randomly sample 2 lines
            sample = random.sample(lines, 2)
            l1, l2 = sample
            vp_candidate = np.cross(l1, l2)
            if np.abs(vp_candidate[2]) < 1e-6:
                continue
            vp_candidate /= vp_candidate[2]

            inliers = []
            for i, line in enumerate(lines):
                # Point-line distance
                d = np.abs(np.dot(line, vp_candidate)) / np.linalg.norm(line[:2])
                if d < threshold:
                    inliers.append(i)

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_vp = vp_candidate

        return np.array(best_vp), np.array(best_inliers)

    def normalize_lines(self, lines: np.ndarray) -> np.ndarray:
        return lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)

    def analyze_motion(self, plot_result: bool = True):
        """
        Analyze motion type (translation or steering) from selected rear light points.
        """
        # Load selected points
        points = np.load(self.selected_points_file)
        if points.ndim != 3 or points.shape[1:] != (2, 2):
            raise ValueError("Selected points must have shape (n, 2, 2)")
        L, R = points[:, 0], points[:, 1]

        # L1, R1, L2, R2 = points
        # L1h, R1h, L2h, R2h = map(self.to_homogeneous, [L1, R1, L2, R2])
        L_h = self.to_homogeneous(L)
        R_h = self.to_homogeneous(R)

        self._logger.debug("Loaded coordinates:")
        # self._logger.debug("  L1: %s", L1)
        # self._logger.debug("  R1: %s", R1)
        # self._logger.debug("  L2: %s", L2)
        # self._logger.debug("  R2: %s", R2)
        for idx, (l, r) in enumerate(zip(L_h, R_h)):
            self._logger.debug(" L%s: %s", idx, l)
            self._logger.debug(" R%s: %s", idx, r)

        # Compute vanishing points
        # Vx, Vy, Vx_h, Vx_k, Vy_h, Vy_k = self.compute_vanishing_points(
        #     L1h, R1h, L2h, R2h, intermediate_results=True
        # )
        # vx_lines hould be L1 R1 and L2 R2
        vx_lines = np.cross(L_h, R_h)
        # vy_lines should be L1 L2 and R1 R2
        vy_lines_left = np.cross(L_h[:-1], L_h[1:])
        vy_lines_right = np.cross(R_h[:-1], R_h[1:])

        vy_lines = np.vstack((vy_lines_left, vy_lines_right))

        vx_lines = self.normalize_lines(vx_lines)
        vy_lines = self.normalize_lines(vy_lines)

        Vx, inliers_x = self.ransac_vanishing_point(list(vx_lines), threshold=0.5)
        Vy, inliers_y = self.ransac_vanishing_point(list(vy_lines), threshold=0.5)

        Vx /= Vx[2]
        Vy /= Vy[2]
        self._logger.debug("Computed vanishing points:")
        self._logger.debug("Vx (vanishing point x): %s", Vx)
        self._logger.debug("Vy (vanishing point y): %s", Vy)
        self._logger.debug("Intermediate results:")
        for idx, (x, y) in enumerate(zip(inliers_x, inliers_y)):
            self._logger.debug("Inlier index for Vx: %s, Vy: %s", x, y)
        

        # Compute vanishing line
        l_inf = np.cross(Vx, Vy)

        self._logger.info(
            "K matrix:\n%s", np.array2string(self.K, precision=3, suppress_small=True)
        )

        # Backproject vanishing points and line into 3D
        Vy_3D, Vx_3D = self.backproject_point_to_3D_ray(np.stack([Vy, Vx]))
        l_inf_3D = self.backproject_line_to_plane_normal(l_inf)

        # Analyze motion type
        dot_product = np.dot(Vx_3D[:2], Vy_3D[:2])
        angle_cos = dot_product / (
            np.linalg.norm(Vx_3D[:2]) * np.linalg.norm(Vy_3D[:2])
        )
        self._logger.info(
            "Dot product of Vx_3D: %s @ Vy_3D: %s = %.4f",
            np.array2string(Vx_3D, precision=3, suppress_small=True),
            np.array2string(Vy_3D, precision=3, suppress_small=True),
            dot_product,
        )
        angle_deg = np.degrees(np.arccos(np.clip(angle_cos, -1.0, 1.0)))
        self._logger.info("Angle between Vx_3D and Vy_3D: %.2f degrees", angle_deg)
        if abs(angle_cos) < 0.1:
            self._logger.info("Result: The car is TRANSLATING FORWARD.")
        else:
            self._logger.info("Result: The car is STEERING (turning).")

        # Plot results if requested
        if plot_result:
            self._plot_results_2D(L_h, R_h, Vx, Vy, vx_lines, vy_lines, l_inf)
            self._plot_results_3D(Vx_3D, Vy_3D, l_inf_3D)

    def _plot_results_2D(
        self,
        L: np.ndarray,
        R: np.ndarray,
        Vx: np.ndarray,
        Vy: np.ndarray,
        vx_lines: np.ndarray = None,
        vy_lines: np.ndarray = None,
        l_inf: np.ndarray = None,
    ) -> None:
        """
        Plot 2D results for vanishing points analysis using arrays of L and R.

        Args:
            L (np.ndarray): Array of left points (n, 3) in homogeneous coordinates.
            R (np.ndarray): Array of right points (n, 3) in homogeneous coordinates.
            Vx (np.ndarray): Homogeneous coordinates of the vanishing point in the x-direction (shape: (3,)).
            Vy (np.ndarray): Homogeneous coordinates of the vanishing point in the y-direction (shape: (3,)).
            vx_lines (np.ndarray, optional): Array of lines for Vx (n, 3).
            vy_lines (np.ndarray, optional): Array of lines for Vy (m, 3).
            l_inf (np.ndarray, optional): Homogeneous representation of the line at infinity (shape: (3,)).
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot all L-R pairs
        for i in range(L.shape[0]):
            ax.plot([L[i, 0], R[i, 0]], [L[i, 1], R[i, 1]], "o-", label=f"Pair {i+1}" if i < 2 else None)

        # Plot vanishing points
        ax.plot(Vx[0], Vx[1], "rx", markersize=10, label="Vx")
        ax.plot(Vy[0], Vy[1], "bx", markersize=10, label="Vy")

        # Plot lines for Vx (L-R lines)
        if vx_lines is not None:
            # Compute x-axis range for plotting lines
            x_vals = np.linspace(
                np.min(np.hstack([L[:, 0], R[:, 0], [Vx[0]], [Vy[0]]])),
                np.max(np.hstack([L[:, 0], R[:, 0], [Vx[0]], [Vy[0]]])),
                500,
            )
            for i, line in enumerate(vx_lines):
                if np.abs(line[1]) > 1e-6:
                    y_vals = -(line[0] * x_vals + line[2]) / line[1]
                    ax.plot(x_vals, y_vals, "g--", alpha=0.5, label="vx_lines" if i == 0 else None)

        # Plot lines for Vy (L-L and R-R lines)
        if vy_lines is not None:
            x_vals = np.linspace(
                np.min(np.hstack([L[:, 0], R[:, 0], [Vx[0]], [Vy[0]]])),
                np.max(np.hstack([L[:, 0], R[:, 0], [Vx[0]], [Vy[0]]])),
                500,
            )
            for i, line in enumerate(vy_lines):
                if np.abs(line[1]) > 1e-6:
                    y_vals = -(line[0] * x_vals + line[2]) / line[1]
                    ax.plot(x_vals, y_vals, "b--", alpha=0.5, label="vy_lines" if i == 0 else None)

        # Plot line at infinity if provided
        if l_inf is not None and np.abs(l_inf[1]) > 1e-6:
            x_vals = np.linspace(
                np.min(np.hstack([L[:, 0], R[:, 0], [Vx[0]], [Vy[0]]])),
                np.max(np.hstack([L[:, 0], R[:, 0], [Vx[0]], [Vy[0]]])),
                500,
            )
            y_vals_l_inf = -(l_inf[0] * x_vals + l_inf[2]) / l_inf[1]
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
        Helper function to plot 3D vectors and visualize their relationships,
        including the XY image plane for better spatial understanding.

        Args:
            Vx (np.ndarray): 3D vector (vanishing point X, backprojected).
            Vy (np.ndarray): 3D vector (vanishing point Y, backprojected).
            l_inf (np.ndarray): 3D vector (vanishing line, backprojected).
        """
        fig_3d = plt.figure(figsize=(8, 8))
        ax_3d = fig_3d.add_subplot(111, projection="3d")

        # Normalize l_inf for visualization
        l_inf_norm = l_inf / np.linalg.norm(l_inf)

        # Plot 3D vectors
        ax_3d.quiver(0, 0, 0, Vx[0], Vx[1], Vx[2], color="r", label="Vx_3D")
        ax_3d.quiver(0, 0, 0, Vy[0], Vy[1], Vy[2], color="b", label="Vy_3D")
        ax_3d.quiver(
            0, 0, 0, l_inf_norm[0], l_inf_norm[1], l_inf_norm[2], color="g", label="l_inf_3D"
        )

        # Plot the XY image plane (z=0)
        plane_size = max(
            np.abs(Vx).max(), np.abs(Vy).max(), np.abs(l_inf_norm).max(), 1
        )
        xx, yy = np.meshgrid(
            np.linspace(-plane_size, plane_size, 10),
            np.linspace(-plane_size, plane_size, 10)
        )
        zz = np.zeros_like(xx)
        ax_3d.plot_surface(
            xx, yy, zz, alpha=0.2, color="gray", label="Image Plane (z=0)"
        )

        # Set plot limits
        max_range = plane_size
        ax_3d.set_xlim([-max_range, max_range])
        ax_3d.set_ylim([-max_range, max_range])
        ax_3d.set_zlim([-max_range, max_range])

        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("3D Visualization of Vx_3D, Vy_3D, l_inf_3D, and Image Plane")
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
        self._logger.info("Calculating camera to plane distance...")
        # Initialize left and right rear light coordinates
        coordinates = np.load(self.selected_points_file)
        L1, R1 = coordinates[0]
        L2, R2 = coordinates[-1]

        # Make the coorindates homogeneous
        L1 = np.array([L1[0], L1[1], 1])
        R1 = np.array([R1[0], R1[1], 1])
        L2 = np.array([L2[0], L2[1], 1])
        R2 = np.array([R2[0], R2[1], 1])

        # Coordinates
        self._logger.debug("Loaded coordinates:")
        self._logger.debug("L1: %s", str(L1))
        self._logger.debug("R1: %s", str(R1))
        self._logger.debug("L2: %s", str(L2))
        self._logger.debug("R2: %s", str(R2))

        # Calculate real world distance between trakced features
        if self.tracked_features == "rear_lights":
            # Load car parameters and estimate rear light distance (6 cm overhand on both sides) (scale it to M instead of MM)
            feature_distance = (
                self.car_params["track_width_mm"]["rear"] + 2 * 60
            ) / 1e3
        elif self.tracked_features == "license_plate":
            # License plate position and parameters are uniform in Italy
            feature_distance = 520 / 1e3
        self._logger.debug(
            "Calculating real life distance between adjacent features: %.2f m",
            feature_distance,
        )

        # Backproject into 3D space
        L1_3D, R1_3D = self.backproject_point_to_3D_ray(np.stack([L1, R1]))
        L2_3D, R2_3D = self.backproject_point_to_3D_ray(np.stack([L2, R2]))

        # Normalize
        L1_3D /= np.linalg.norm(L1_3D)
        R1_3D /= np.linalg.norm(R1_3D)
        L2_3D /= np.linalg.norm(L2_3D)
        R2_3D /= np.linalg.norm(R2_3D)

        self._logger.debug("Backprojecting coordinates to 3D coordinates:")
        self._logger.debug("L1 3D: %s", str(L1_3D))
        self._logger.debug("R1 3D: %s", str(R1_3D))
        self._logger.debug("L2 3D: %s", str(L2_3D))
        self._logger.debug("R2 3D: %s", str(R2_3D))

        # Estimate scale
        scale = self.estimate_camera_distance_cosine(L1_3D, R1_3D, feature_distance)
        self._logger.info("The scale of the first triangle: %.2f", scale)
        scale2 = self.estimate_camera_distance_cosine(L2_3D, R2_3D, feature_distance)
        self._logger.info("The scale of the second triangle: %.2f", scale2)

        if plot_results:
            self._plot_cosine_triangle(L1_3D, R1_3D, scale)
            self._plot_cosine_triangle(L2_3D, R2_3D, scale2)

    def _plot_cosine_triangle(
        self, ray1: np.ndarray, ray2: np.ndarray, scale: float
    ) -> None:
        """
        Plot the triangle formed by the camera and two scaled rays using the law of cosines,
        and visualize the XY plane for spatial reference.

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

        # Plot the XY plane (z=0)
        max_range = np.max(np.abs([P1, P2, origin])) * 1.2
        xx, yy = np.meshgrid(
            np.linspace(-max_range, max_range, 10),
            np.linspace(-max_range, max_range, 10)
        )
        zz = np.zeros_like(xx)
        ax.plot_surface(
            xx, yy, zz, alpha=0.15, color="gray", label="XY Plane"
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
