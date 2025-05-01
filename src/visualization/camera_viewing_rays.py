import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

# Camera intrinsics (from user)
# K = np.array([
#     [16380.,     0., 2016.],
#     [    0., 16380., 1512.],
#     [    0.,     0.,    1.]
# ])
fs = cv2.FileStorage("config/camera_intrinsics.yaml", cv2.FILE_STORAGE_READ)
K = fs.getNode("K").mat()
fs.release()
print(K)

K_inv = np.linalg.inv(K)

# Define a few sample image points in homogeneous coordinates (center and corners)
image_points = np.array([
    [2016, 1512, 1],       # center
    [0, 0, 1],             # top-left
    [4032, 0, 1],          # top-right
    [0, 3024, 1],          # bottom-left
    [4032, 3024, 1],       # bottom-right
    [2016, 0, 1],          # middle top
    [2016, 3024, 1],       # middle bottom
])

# Backproject each point to a ray
rays = []
for p in image_points:
    ray = K_inv @ p
    ray /= np.linalg.norm(ray)
    rays.append(ray)

rays = np.array(rays)

# Compute horizontal and vertical field of view (FOV)
center_ray = rays[0]
left_ray = rays[1]  # top-left
right_ray = rays[2]  # top-right

top_ray = rays[1]    # top-left
bottom_ray = rays[3] # bottom-left

# Horizontal FOV
cos_h = np.dot(left_ray, right_ray)
fov_h_rad = np.arccos(np.clip(cos_h, -1.0, 1.0))
fov_h_deg = np.degrees(fov_h_rad)

# Vertical FOV
cos_v = np.dot(top_ray, bottom_ray)
fov_v_rad = np.arccos(np.clip(cos_v, -1.0, 1.0))
fov_v_deg = np.degrees(fov_v_rad)

print(f"Estimated Horizontal FOV: {fov_h_deg:.2f} degrees")
print(f"Estimated Vertical FOV: {fov_v_deg:.2f} degrees")

# Plot the rays in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

origin = np.zeros((3,))
for i, ray in enumerate(rays):
    ax.quiver(*origin, *ray, length=1.0, normalize=True, label=f'Ray {i+1}')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 1.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Backprojected 3D Rays from Image Points')
ax.legend()
plt.show()
