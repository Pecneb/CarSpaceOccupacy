import numpy as np
import matplotlib.pyplot as plt
import fire
import os

def manual_select_frames(trajectory_file, output_file=None, plot_selected=True):
    # Load the tracked points
    trajectories = np.load(trajectory_file)  # shape (2, num_frames, 2)
    light1 = trajectories[0]
    light2 = trajectories[1]
    num_frames = light1.shape[0]

    # Average points between lights to make click selection easier
    middle_points = (light1 + light2) / 2

    selected_frames = []

    def onclick(event):
        if event.inaxes:
            x_click, y_click = event.xdata, event.ydata
            # Find nearest frame
            distances = np.linalg.norm(middle_points - np.array([x_click, y_click]), axis=1)
            closest_frame = np.argmin(distances)
            selected_frames.append(closest_frame)
            print(f"Selected Frame {closest_frame}")
            if len(selected_frames) == 2:
                plt.close()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(light1[:, 0], light1[:, 1], 'r-', label='Rear Light 1')
    ax.plot(light2[:, 0], light2[:, 1], 'b-', label='Rear Light 2')
    ax.plot(middle_points[:, 0], middle_points[:, 1], 'ko', markersize=3, label='Middle Points')
    ax.invert_yaxis()
    ax.set_title("Click Two Points to Select Two Frames")
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(selected_frames) != 2:
        print("Error: You must select exactly two frames.")
        return None
    else:
        selected_frames.sort()
        print(f"Selected frames: {selected_frames}")

        if output_file:
            # Extract and save the 4 points: L1, R1, L2, R2
            L1 = light1[selected_frames[0]]
            R1 = light2[selected_frames[0]]
            L2 = light1[selected_frames[1]]
            R2 = light2[selected_frames[1]]

            selected_points = np.array([L1, R1, L2, R2])
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.save(output_file, selected_points)
            print(f"Saved selected points to {output_file}")

            if plot_selected:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(light1[:, 0], light1[:, 1], 'r-', label='Rear Light 1')
                ax.plot(light2[:, 0], light2[:, 1], 'b-', label='Rear Light 2')
                ax.plot(middle_points[:, 0], middle_points[:, 1], 'ko', markersize=3, label='Middle Points')
                ax.plot([L1[0], R1[0]], [L1[1], R1[1]], 'g-o', label='Light Segment Frame 1')
                ax.plot([L2[0], R2[0]], [L2[1], R2[1]], 'm-o', label='Light Segment Frame 2')
                ax.invert_yaxis()
                ax.set_title("Selected Light Segments")
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Y (pixels)')
                ax.legend()
                plt.show()

        return selected_frames

if __name__ == "__main__":
    fire.Fire(manual_select_frames)
