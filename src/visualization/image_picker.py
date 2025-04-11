import os
import cv2
import fire


def extract_images(
    video_path: str,
    output_dir: str,
    frame_interval: int,
    start_time: int,
    end_time: int,
):
    """Extract images from video at specified intervals and time range (basically like get a clip from the source video and save and image every x frames).

    Args:
        video_path (str): Path to source video.
        output_dir (str): Path to output directory where images will be saved.
        frame_interval (int): Give the frequency at which the images will be taken. (Like save an image every 3 frames)
        start_time (int): The start time in seconds.
        end_time (int): The end time in seconds.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Validate time range
    if start_time < 0 or end_time > duration or start_time >= end_time:
        print("Error: Invalid time range.")
        cap.release()
        return

    # Calculate frame range
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames
    frame_count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > end_frame:
            break

        if (
            start_frame <= frame_count <= end_frame
            and frame_count % frame_interval == 0
        ):
            output_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extraction complete. {saved_count} images saved to {output_dir}.")


if __name__ == "__main__":
    fire.Fire(extract_images)
