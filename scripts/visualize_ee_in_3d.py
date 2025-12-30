import argparse
from pathlib import Path
from so101_ik_fk.utils.visualization import create_trajectory_gif
from so101_ik_fk.utils.data import download_and_load_data, extract_joint_values
from so101_ik_fk.lib.so101_kinematics import SO101ForwardKinematics

# Global variables/Hyperparameters
DATASET_REPO_ID = "sapanostic/so101_offline_eval"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument("--out", type=str, default="outputs/episode_trajectory.gif")
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print(f"Loading dataset: {DATASET_REPO_ID}...")
    dataset = download_and_load_data(DATASET_REPO_ID)

    # Standard HF dataset doesn't have num_episodes, but we can check unique indices
    unique_episodes = set(dataset['episode_index'])
    if args.episode not in unique_episodes:
        print(f"Error: Episode index {args.episode} not found in dataset")
        return

    # 2. Extract Joint Values
    print(f"Processing Episode {args.episode}...")

    # Get joint names order from kinematics to ensure correct mapping
    kinematics = SO101ForwardKinematics()
    joint_names = kinematics.kinematics.joint_names

    joint_values = extract_joint_values(dataset, args.episode, joint_names_filter=joint_names)

    # 3. Create Visualization
    create_trajectory_gif(
        joint_values=joint_values,
        output_gif=str(output_path),
        output_png=str(output_path.with_suffix(".png")),
        image_key=f"Episode {args.episode}"
    )

if __name__ == "__main__":
    main()
