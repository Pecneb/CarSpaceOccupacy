import numpy as np
import matplotlib.pyplot as plt
import fire


def plot_trajectories(npy_path, save_path=None):
    # Load tracked trajectories
    trajectories = np.load(npy_path)

    if trajectories.shape[0] != 2:
        print("Error: Expected two trajectories (for two points).")
        return

    light1 = np.array(trajectories[0])  # shape (num_frames, 2)
    light2 = np.array(trajectories[1])

    plt.figure(figsize=(8, 6))
    plt.plot(light1[:, 0], light1[:, 1], 'r-', label='Rear Light 1')
    plt.plot(light2[:, 0], light2[:, 1], 'b-', label='Rear Light 2')
    plt.scatter(light1[0, 0], light1[0, 1], c='r', marker='o', label='Start Light 1')
    plt.scatter(light2[0, 0], light2[0, 1], c='b', marker='o', label='Start Light 2')

    plt.gca().invert_yaxis()  # because image coordinates have (0,0) at top-left
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.title('Tracked Rear Lights Trajectories')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    fire.Fire(plot_trajectories)
