import cv2
import numpy as np
import glob

# === Parameters ===
checkerboard_size = (9, 7)  # number of inner corners
square_size = 25  # mm

# === Prepare 3D object points (real world coordinates) ===
objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# === Load checkerboard images ===
images = glob.glob('data/calibration/*.jpg')  # your folder with checkerboard images
print(images)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # Optional: draw and show
        cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# === Calibrate ===
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# === Save results ===
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)

# Optional: save to file
np.savez("calibration_data.npz", camera_matrix=mtx, dist_coeffs=dist)
