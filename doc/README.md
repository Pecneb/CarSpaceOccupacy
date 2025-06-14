# Workflow to Track Rear Lights and Select Frames for 3D Reconstruction

This document summarizes the steps and script usage needed to arrive at the point where 4 key points (L1, R1, L2, R2) are extracted for further 3D reconstruction analysis.

## Step 1: Track Rear Lights

### Script: `track_rear_lights.py`

**Usage:**
```bash
python feature_tracking.py --video_path path_to_video.mp4 --output_path tracked_rear_lights.npy --start_frame 0 --display True
```

**Description:**
- Load the video.
- Manually click on the two rear lights in the selected starting frame.
- Track the two lights across all frames using Lucas-Kanade Optical Flow.
- Save the tracked points into `tracked_rear_lights.npy`.

---

## Step 2: Visualize Trajectories

### Script: `plot_tracked_trajectory.py`

**Usage:**
```bash
python visualize_track.py --trajectory_file tracked_rear_lights.npy
```

**Description:**
- Load and plot the trajectories of the two rear lights.
- Visual inspection to understand the motion and trajectory quality.

---

## Step 3: Select Two Frames for Geometric Analysis

### Script: `plot_tracked_trajectory.py` (with `manual_select_frames` function)

**Usage:**
```bash
python select_points.py --trajectory_file tracked_rear_lights.npy --output_file selected_points.npy
```

**Description:**
- Click two points along the trajectory plot to select two frames.
- The two selected frames correspond to two "light segments".
- Extract and save the 4 points (L1, R1, L2, R2) into `selected_points.npy`.
- Automatically visualize and confirm the selection.

---

## Step 4: Analyze Motion Type (Translation or Steering)

### Script: `analyze_motion.py`

**Usage:**
```bash
python3 src/analysis/motion_analysis.py --selected_points_file=data/tracks/selected_points.npy -K=config/camera_intrinsics.yaml analyze_motion
```

**Description:**
- Load the selected 4 points (L1, R1, L2, R2).
- Load the camera intrinsic matrix from the OpenCV-style YAML file.
- Compute the light segments (lines h and k).
- Calculate vanishing points (Vx and Vy) following the professor's method.
- Analyze the cosine of the angle between the 3D directions.
- Determine if the car is translating forward or steering.
- Optionally plot the light segments and vanishing points for visualization.

---

## Step 5: Compute the distance of camera from rear lights

### Script `analyze_motion.py`

**Usage:**
```bash
python motion_analysis.py --selected_points_file=data/tracks/selected_points.npy --K_file=config/camera_intrinsics.yaml --car_params_file=config/black_audi_a3_sportback.yaml camera_to_plane_distance
```

**Description:**
- Load the selected 4 points from the npy file.
- We only need 1 pair of lights now.
- Load the yaml file containing the car params.
- Compute the distance using the theorem of cosines.
- Print the distance in meters.
- Plot the triangle in 3D.

# Files and Artifacts

- `tracked_rear_lights.npy` : Tracked positions of rear lights across frames.
- `selected_points.npy` : 4 selected points (L1, R1, L2, R2) for reconstruction.
- `camera_intrinsics.yaml` : Camera intrinsic matrix saved using OpenCV FileStorage.

# Next Steps

Proceed to reconstruct the true 3D trajectory using real-world distance between the rear lights and the theory of cosines.
