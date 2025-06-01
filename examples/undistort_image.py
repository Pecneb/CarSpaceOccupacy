import cv2
import numpy as np
import fire
from pathlib import Path


def main(calibration_data_path: str, image_path: str):
    # Check existence of calibration data path
    calibration_data_path = Path(calibration_data_path)
    if not calibration_data_path.exists():
        raise FileNotFoundError(
            f"Calibration data file was not found: {str(calibration_data_path)}"
        )

    # Load from file
    data = np.load(str(calibration_data_path))

    # Extract camera matrix and distortion coefficients
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    # Check the existence of the image
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image was not found: {str(image_path)}")

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    cv2.imshow("Undistorted", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fire.Fire(main)
