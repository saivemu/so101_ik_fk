import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from pathlib import Path
from ..lib.so101_kinematics import SO101ForwardKinematics

def create_trajectory_gif(
    joint_values: np.ndarray,
    output_gif: str,
    output_png: str | None = None,
    urdf_path: str | None = None,
    image_key: str = "Robot Visualization"
):
    """
    Create a trajectory GIF from joint values.

    Args:
        joint_values: Array of joint values (N, 6) or (N, 5).
        output_gif: Path to save the output GIF.
        output_png: Path to save the final frame as PNG (optional).
        urdf_path: Path to the URDF file. If None, uses default from kinematics class.
        image_key: Title for the plot.
    """

    # Load Kinematics
    # The kinematics class handles default URDF path internally
    kinematics = SO101ForwardKinematics(urdf_path=urdf_path)

    # Let's pre-calculate EE positions first
    # Assume input is already ordered for the robot (first 5 or 6 joints)
    # The solver expects [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]

    if joint_values.shape[1] > 5:
        # Take first 5 joints if more are provided (e.g. gripper included)
        ordered_joint_values = joint_values[:, :5].astype(np.float64)
    else:
        ordered_joint_values = joint_values.astype(np.float64)

    ee_positions = []
    for i in range(len(ordered_joint_values)):
        pos = kinematics.get_ee_position(ordered_joint_values[i])
        ee_positions.append(pos)
    ee_positions = np.array(ee_positions)

    # Setup Animation
    print("Setting up animation...")
    fig = plt.figure(figsize=(10, 8))
    # Single 3D subplot since we decoupled dataset images
    ax_3d = fig.add_subplot(111, projection='3d')
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title(f'End Effector Trajectory - {image_key}')

    # Initialize plots
    # Full trajectory line (faint)
    ax_3d.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 'b--', alpha=0.3, label='Full Path')

    # Current path line (solid)
    line, = ax_3d.plot([], [], [], 'b-', linewidth=2, label='Traversed')
    # Current point
    point, = ax_3d.plot([], [], [], 'ro', markersize=5)

    # Set fixed limits for 3D plot
    margin = 0.05
    ax_3d.set_xlim(ee_positions[:, 0].min() - margin, ee_positions[:, 0].max() + margin)
    ax_3d.set_ylim(ee_positions[:, 1].min() - margin, ee_positions[:, 1].max() + margin)
    ax_3d.set_zlim(ee_positions[:, 2].min() - margin, ee_positions[:, 2].max() + margin)
    ax_3d.legend()

    def update(frame_idx):
        # Update trajectory
        # Plot path up to current frame
        current_path = ee_positions[:frame_idx+1]
        line.set_data(current_path[:, 0], current_path[:, 1])
        line.set_3d_properties(current_path[:, 2])

        # Update current point
        point.set_data([ee_positions[frame_idx, 0]], [ee_positions[frame_idx, 1]])
        point.set_3d_properties([ee_positions[frame_idx, 2]])

        return line, point

    print(f"Creating animation with {len(ee_positions)} frames...")
    ani = FuncAnimation(fig, update, frames=len(ee_positions), interval=50, blit=False)

    if output_png:
        print(f"Saving final frame image to {output_png}...")
        update(len(ee_positions) - 1)
        fig.savefig(output_png)

    print(f"Saving GIF to {output_gif}...")
    ani.save(output_gif, writer='pillow', fps=15)
    print("Done!")
