# Workflow to Track Rear Lights and Select Frames for 3D Reconstruction

This document summarizes the steps and script usage needed to arrive at the point where 4 key points (L1, R1, L2, R2) are extracted for further 3D reconstruction analysis.

## Step 1: Track Rear Lights

### Script: `track_rear_lights.py`

**Usage:**
```bash
python track_rear_lights.py --video_path path_to_video.mp4 --output_path tracked_rear_lights.npy --start_frame 0 --display True
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
python plot_tracked_trajectory.py --trajectory_file tracked_rear_lights.npy
```

**Description:**
- Load and plot the trajectories of the two rear lights.
- Visual inspection to understand the motion and trajectory quality.

---

## Step 3: Select Two Frames for Geometric Analysis

### Script: `plot_tracked_trajectory.py` (with `manual_select_frames` function)

**Usage:**
```bash
python plot_tracked_trajectory.py --trajectory_file tracked_rear_lights.npy --output_file selected_points.npy
```

**Description:**
- Click two points along the trajectory plot to select two frames.
- The two selected frames correspond to two "light segments".
- Extract and save the 4 points (L1, R1, L2, R2) into `selected_points.npy`.
- Automatically visualize and confirm the selection.

---

# Files and Artifacts

- `tracked_rear_lights.npy` : Tracked positions of rear lights across frames.
- `selected_points.npy` : 4 selected points (L1, R1, L2, R2) for reconstruction.

# Next Steps

Proceed to geometric processing: calculate vanishing points, decide translation vs steering, estimate plane π.

