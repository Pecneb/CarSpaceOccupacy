# CarSpaceOccupacy
IACV Project implementation, which encapsulates car space occupacy analysis with image processing techniques.

### **Project Description: Car Motion and Space Occupancy Analysis in the Dark**

#### **Overview**
This project focuses on developing a computer vision system that analyzes a video of a moving vehicle taken by a fixed camera. The primary goal is to determine the **space occupied by the vehicle** in each frame while accounting for different motion conditions, including forward translation and steering at constant curvature. The system utilizes **camera calibration data** and a **simplified vehicle model** to estimate the vehicle’s 3D position on the road.

#### **Key Components**
1. **Camera Calibration:** The intrinsic parameters of the camera (K-matrix) are provided and used for geometric calculations.
2. **Vehicle Model:** A simplified model of the car includes:
   - Car width and length.
   - Distance between the rear lights.
   - Height of rear lights from the ground.
   - Distance of the rear lights from the back of the car.

#### **Assumptions**
- The moving car has a **vertical symmetry plane**.
- Two **symmetric rear lights** are visible.
- The camera observes the **back of the vehicle**.
- The road is **locally planar**.
- Between consecutive frames, the car either:
  - Translates **forwards**.
  - **Steers** with a constant curvature.

#### **System Operation**
##### **Offline Steps**
1. **Camera Calibration** to retrieve intrinsic parameters.
2. **Vehicle Model Extraction**, including dimensions and light placements.

##### **Online Steps (Per Frame Analysis)**
1. **Feature Extraction:** Identify the two symmetric rear lights (or distinct features on them).
2. **Geometric Analysis:** Determine the **3D position** of the light segment and its containing plane (π).
3. **Space Occupancy Estimation:** Using the car model and computed positions, estimate the **occupied space** on the road.

#### **Geometric Principles Used**
1. **Forward Translation:**
   - If the car is moving straight, the **first and second light segments** form a **rectangle**.
   - The system finds **vanishing points** (Vx, Vy) and the **vanishing line (l)**.
   - 3D localization of the light segments is achieved using geometric constraints (e.g., **theorem of cosines**).

2. **Steering with Constant Curvature:**
   - The light segment rotates about a **center of rotation (C)**.
   - The method estimates **vanishing points** and **perpendicular constraints** to determine motion curvature.

#### **Challenges & Solutions**
1. **Poor Perspective (Parallel Viewing Rays)**
   - The method requires **sufficient perspective effects** to work.
   - If perspective is weak, additional **non-coplanar symmetric elements** are used.
   - The system estimates the car’s **pose** using an **iterative refinement** approach.

2. **Car Localization Methods**
   - **Standard Homography-Based Localization:** Uses **license plate corners and light features**.
   - **Nighttime Localization from Image Pairs:** Uses **rear lights** to estimate motion.
   - **Localization with Out-of-Plane Features:** When poor perspective makes angle estimation unreliable, the **difference in symmetric elements on separate planes** is used.

3. **Iterative Refinement for Pose Estimation**
   - **Refines estimates** of direction and position.
   - Updates **vanishing points** and **horizontal vanishing lines**.
   - Uses **camera calibration matrix** to improve estimates.

### **Conclusion**
The system provides an efficient **3D reconstruction of vehicle motion** based on **video analysis**, ensuring accurate **space occupancy estimation** in **low-light conditions**. This has potential applications in **traffic monitoring, autonomous vehicle navigation, and night-time vehicle tracking**.


Since we are implementing the **space occupancy analysis system in Python**, it's best to follow a **modular, scalable project structure** that allows clean separation between components like camera calibration, feature extraction, geometric computations, and visualization.

Directory structure guide:

```
space_occupancy/
│
├── data/                         # All raw video inputs, calibration data, and sample frames
│   ├── videos/
│   ├── calibration/
│   └── sample_frames/
│
├── configs/                      # Configuration files (e.g., camera matrix, car model)
│   ├── camera_intrinsics.yaml
│   └── car_model.yaml
│
├── src/                          # All source code modules
│   ├── __init__.py
│   ├── main.py                   # Entry point
│   ├── camera/                   # Camera calibration & projection logic
│   │   ├── __init__.py
│   │   ├── calibration.py
│   │   └── projection.py
│   │
│   ├── detection/                # Feature and light detection logic
│   │   ├── __init__.py
│   │   ├── light_detector.py
│   │   └── feature_tracking.py
│   │
│   ├── geometry/                 # 3D geometry, vanishing points, pose estimation
│   │   ├── __init__.py
│   │   ├── space_occupancy.py
│   │   ├── plane_estimation.py
│   │   └── vanishing_points.py
│   │
│   ├── utils/                    # Helper functions (I/O, drawing, math)
│   │   ├── __init__.py
│   │   ├── io_utils.py
│   │   ├── drawing.py
│   │   └── math_utils.py
│   │
│   └── visualization/           # Plotting, annotated videos, image output
│       ├── __init__.py
│       ├── viewer.py
│       └── video_writer.py
│
├── notebooks/                   # Jupyter notebooks for testing and visualization
│   └── debug_pose_estimation.ipynb
│
├── tests/                       # Unit and integration tests
│   ├── test_geometry.py
│   └── test_light_detection.py
│
├── results/                     # Outputs: 3D reconstructions, videos, logs
│   ├── plots/
│   └── annotated_videos/
│
├── requirements.txt             # Python dependencies
├── README.md
└── .gitignore
```

---

### 📌 Suggestions for Implementation
- Use **OpenCV** (`cv2`) for image I/O and basic feature detection.
- Use **NumPy** for matrix math and 3D operations.
- Consider **SciPy** or **OpenCV's solvePnP** for pose estimation.
- Store camera intrinsics and car model parameters in `.yaml` or `.json` files under `configs/`.
- Add **logging** and debugging support (e.g., timestamps, feature match quality).

Would you like me to generate a `requirements.txt` and a sample config file (like `camera_intrinsics.yaml`) to start from?


For the following photo we follow the steps below to extract useful information for the algorithm. Our target is the red kia sportage.

![Red Kia Sportage](/data/sample_frames/red_kia_sportage/frame_175.jpg)

---

## ✅ **1. Assign Real-World Coordinates to Key Points**
Using the **physical dimensions**:
- Rear lights are symmetric, so choose two reference points on them — e.g., centers of each light.
- Approximate the **rear light distance** from the specs:
  - If rear track is 1636 mm and light centers are slightly inset, estimate ~1500 mm between rear light centers.
- Assume the rear light height is roughly **halfway up the car**, say ~800–900 mm from the ground.
- Car **length = 4480 mm**, so the **rear bumper to front bumper span** is known if needed.

---

## 🧠 **2. Extract Corresponding Image Points**
From the uploaded image:
- Identify and annotate pixel coordinates of:
  - Rear-left light center → L
  - Rear-right light center → R
  - Bottom of wheels (for ground contact)
  - Roofline midpoint (for vertical context)
- Optional: license plate corners or edges of the bumper for extra symmetry cues.

---

## 🔁 **3. Back-Project to 3D**
Assuming you have the **camera calibration matrix K**, for each image point \( p \) (in homogeneous coordinates):
```math
\text{Ray direction} = K^{-1} p
```
You get the **3D ray** from the camera through the light points. You need at least **one depth estimate** to localize them in 3D — here's where your prior model dimensions help.

---

## 📐 **4. Estimate the Ground Plane (π)**
Using:
- The 3D directions of the rear lights (from rays)
- Their known real-world separation
- The assumption that the rear lights lie on a vertical plane
You can:
- Estimate π using triangulation (for car translating case), or
- Use the **vanishing points** method from the slides (Space Occupancy PDF)

---

## 🔍 **5. Compute Space Occupied**
Once π and the 3D rear points are known:
- Reconstruct a rectangle or quadrilateral approximating the vehicle's footprint.
- This is your **occupied space on the road** in each frame.

---

## ⚙️ Optional Enhancements
- Use **Symmetric Feature Localization** from the slides for better pose estimation under low perspective.
- Use multiple frames to track light motion and deduce forward/curved motion (as outlined in the “Car translating” vs “Car steering” cases).

---