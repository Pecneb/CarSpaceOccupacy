import cv2
import numpy as np
import fire
import os
from typing import Optional
from pathlib import Path


# --- Step 1: Define the main function ---
def track_lights(
    video_path: str,
    output_path: str = "data/tracks/tracked_rear_lights.npy",
    display: bool = True,
    start_frame: int = 0,
    calibration_data_path: Optional[str] = None,
):
    # --- Load calibartion matrices if give ---
    if calibration_data_path:
        calibration_data = Path(calibration_data_path)
        if not calibration_data.exists():
            raise FileNotFoundError(
                f"Given source file path for calibration matrices was not found: {calibration_data}"
            )
        calibration_matrices = np.load(str(calibration_data))
        camera_matrix = calibration_matrices["camera_matrix"]
        dist_coeffs = calibration_matrices["dist_coeffs"]
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", dist_coeffs)

    # --- Load video ---
    cap = cv2.VideoCapture(video_path)

    if calibration_data_path:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        # Get new camera matrix
        newcameramtx, rio = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return

    # Undistort the frame
    if calibration_data_path:
        first_frame = cv2.undistort(
            first_frame, camera_matrix, dist_coeffs, None, newcameramtx
        )

    # --- Manually select two points ---
    selected_points = []

    if display:
        # Mouse callback to select points
        def select_point(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(selected_points) < 2:
                    selected_points.append([x, y])
                    print(f"Selected point {len(selected_points)}: ({x}, {y})")

        cv2.namedWindow("Select Lights")
        cv2.setMouseCallback("Select Lights", select_point)

        print("Click on the two rear lights in the first frame.")

        temp_frame = first_frame.copy()
        while len(selected_points) < 2:
            for pt in selected_points:
                cv2.circle(temp_frame, tuple(pt), 5, (0, 255, 0), -1)
            cv2.imshow("Select Lights", temp_frame)
            pressed_key = cv2.waitKey(1)
            if pressed_key & 0xFF == 27:  # ESC to exit
                break
            elif pressed_key == ord("d"):
                ret, temp_frame = cap.read()
                if not ret:
                    exit(0)

        cv2.destroyWindow("Select Lights")
    else:
        print(
            "Display is disabled. Please modify the script to input points manually if needed."
        )
        return

    # Prepare points for tracking
    prev_pts = np.array(selected_points, dtype=np.float32).reshape(-1, 1, 2)

    # Lucas–Kanade Optical Flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # --- Track points ---
    frame_idx = 0
    trajectories = [[], []]  # One list per light

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if calibration_data_path:
            frame = cv2.undistort(
                frame, camera_matrix, dist_coeffs, None, newcameramtx
            )

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            first_frame, frame, prev_pts, None, **lk_params
        )

        if next_pts is not None and status.sum() == 2:
            for i, pt in enumerate(next_pts):
                x, y = pt.ravel()
                trajectories[i].append((x, y))
                if display:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        prev_pts = next_pts.reshape(-1, 1, 2)
        first_frame = frame.copy()

        if display:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()

    # --- Save or process trajectories ---
    trajectories = np.array(trajectories)  # shape: (2, num_frames, 2)
    trajectories[:, :, 1] = frame.shape[0] - trajectories[:, :, 1]

    print("Tracking finished.")
    print("Trajectories shape:", trajectories.shape)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, trajectories)
    print(f"Saved tracked points to '{output_path}'")


# --- Entry point ---
if __name__ == "__main__":
    fire.Fire(track_lights)
