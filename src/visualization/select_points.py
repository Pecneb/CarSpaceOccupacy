import numpy as np
import matplotlib.pyplot as plt
import fire
import os

def manual_select_frames(trajectory_file, number_of_frames=5, output_file=None, plot_selected=True):
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
            if len(selected_frames) == number_of_frames:
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
    ax.invert_yaxis()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(selected_frames) != number_of_frames:
        print(f"Error: You must select exactly {number_of_frames} frames.")
        return None
    else:
        selected_frames.sort()
        print(f"Selected frames: {selected_frames}")

        if output_file:
            # Extract and save the 4 points: L1, R1, L2, R2
            # L1 = light1[selected_frames[0]]
            # R1 = light2[selected_frames[0]]
            # L2 = light1[selected_frames[1]]
            # R2 = light2[selected_frames[1]]

            L = np.array(light1[selected_frames])
            R = np.array(light2[selected_frames])

            # selected_points = np.array([L1, R1, L2, R2])

            selected_points = np.stack((L, R), axis=1)
            print(f"Selected coordinates: {selected_points}")
            print(f"Shape of selected coordinates: {selected_points.shape}")

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            np.save(output_file, selected_points)
            print(f"Saved selected points to {output_file}")

            if plot_selected:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(light1[:, 0], light1[:, 1], 'r-', label='Rear Light 1')
                ax.plot(light2[:, 0], light2[:, 1], 'b-', label='Rear Light 2')
                ax.plot(middle_points[:, 0], middle_points[:, 1], 'ko', markersize=3, label='Middle Points')
                colors = ['g', 'm', 'c', 'y', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
                for idx, frame in enumerate(selected_frames):
                    color = colors[idx % len(colors)]
                    ax.plot([light1[frame, 0], light2[frame, 0]],
                            [light1[frame, 1], light2[frame, 1]],
                            marker='o', color=color, label=f'Light Segment Frame {frame}')
                ax.invert_yaxis()
                ax.set_title("Selected Light Segments")
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Y (pixels)')
                ax.legend()
                ax.invert_yaxis()
                plt.show()

        return selected_frames

if __name__ == "__main__":
    fire.Fire(manual_select_frames)
