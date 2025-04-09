import exifread
import numpy as np
import cv2


def estimate_intrinsics_from_exif(image_path, sensor_width_mm=6.4):
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f)

    # Try to extract focal length (in mm)
    focal_tag = tags.get("EXIF FocalLength")
    focal_length_mm = (
        float(focal_tag.values[0].num) / focal_tag.values[0].den if focal_tag else 26.0
    )  # iPhone wide default

    # Try to get image dimensions
    img_width = int(tags.get("EXIF ExifImageWidth", 4032))
    img_height = int(tags.get("EXIF ExifImageLength", 3024))

    print(f"Image size: {img_width}x{img_height}")
    print(f"Focal Length: {focal_length_mm} mm")

    # Focal length in pixels (horizontal)
    fx = (focal_length_mm / sensor_width_mm) * img_width
    fy = fx  # assuming square pixels

    cx = img_width / 2
    cy = img_height / 2

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    print("\nEstimated Intrinsic Matrix (K):")
    print(K)
    return K


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("frame", type=str, default="frame.jpg")
    args = parser.parse_args()
    path_to_frame = args.frame
    path_to_frame = Path(path_to_frame)
    if not path_to_frame.exists():
        print(f"File {path_to_frame} was not found!")
        exit(-1)
    # Run on extracted frame
    K = estimate_intrinsics_from_exif(str(path_to_frame.absolute()))

    # Save to YAML file
    fs: cv2.FileStorage = cv2.FileStorage("camera_intrinsics.yaml", cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    # fs.write("distortion", dist)
    fs.release()
    print("Saved camera parameters to YAML.")
