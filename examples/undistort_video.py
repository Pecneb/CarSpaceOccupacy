import cv2
import numpy as np
import fire
from pathlib import Path
from time import time


def main(src: str, calibration_data_path: str):
    """Undistort video with camera intrinsic parameters.

    Args:
        src (str): Path to source video.
        k_matrix (str): Path to K-Matrix with intrinsic parameters.
    """
    # Check existence of given files
    source = Path(src)
    if not source.exists():
        raise FileNotFoundError(f"Give source file path was not found {source}")
    calibration_data = Path(calibration_data_path)
    if not calibration_data.exists():
        raise FileNotFoundError(
            f"Give source file path was not found {calibration_data}"
        )

    # Open Video
    cap = cv2.VideoCapture(str(source))

    # get framerate
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(round(1e3/fps))

    # get frame width and height
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # Get calibration parameters
    # Load from file
    data = np.load(str(calibration_data_path))

    # Extract camera matrix and distortion coefficients
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    newcameramtx, rio = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # Play video
    while True:
        t_start = time()
        # Retrieve next frame
        ret, frame = cap.read()
        if ret is None:
            break

        # Display original frame
        cv2.imshow("Original Video Frame", frame)

        # Undistrort frame with camera calibration parameters
        undistorted_frame = cv2.undistort(
            frame, camera_matrix, dist_coeffs, None, newcameramtx
        )
        # Display undistorted frame
        cv2.imshow("Undistorted Video Frame", undistorted_frame)

        if cv2.waitKey(1) == ord("q"):
            break
        # print(f"FPS: {1/(time()-t_start)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fire.Fire(main)
