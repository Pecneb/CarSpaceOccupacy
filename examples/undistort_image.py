import cv2
import numpy as np

# Load from file
data = np.load("calibration_data.npz")

# Extract camera matrix and distortion coefficients
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)


img = cv2.imread("your_image.jpg")
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
