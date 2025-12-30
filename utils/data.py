import torch
import numpy as np
import logging

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

def download_and_load_data(repo_id: str, split: str = "train"):
    """
    Download and load dataset from Hugging Face Hub.

    Args:
        repo_id: Hugging Face dataset repository ID.
        split: Dataset split to load.

    Returns:
        The loaded dataset.
    """
    if load_dataset is None:
        raise ImportError("The 'datasets' library is required to load data. Please install it with `pip install datasets`.")

    print(f"Loading dataset from HF Hub: {repo_id}")
    dataset = load_dataset(repo_id, split=split)
    return dataset

def extract_joint_values(dataset, episode_index: int, joint_names_filter: list[str] | None = None):
    """
    Extract joint values for a specific episode from a Hugging Face dataset.

    Args:
        dataset: The loaded HF dataset.
        episode_index: Index of the episode to extract.
        joint_names_filter: Optional list of joint names to filter/order the output.

    Returns:
        np.ndarray: Array of joint values.
    """
    # Filter dataset for the specific episode
    # Note: This assumes the dataset has 'episode_index' column
    # Ideally, we should use efficient filtering, but for simple use:
    episode_data = dataset.filter(lambda x: x['episode_index'] == episode_index)

    if len(episode_data) == 0:
        raise ValueError(f"No data found for episode index {episode_index}")

    # Extract observation.state
    # HF datasets usually return lists or formatted items
    # We want to stack them into a numpy array

    obs_state = episode_data['observation.state']

    if isinstance(obs_state[0], torch.Tensor):
        joint_values = torch.stack(obs_state).numpy()
    elif isinstance(obs_state[0], list):
        joint_values = np.array(obs_state)
    else:
        joint_values = np.array(obs_state)

    # If filter is provided, we need to map indices
    # This assumes 'observation.state' in dataset has a 'names' feature or similar metadata
    # But standard HF dataset loading might not preserve custom feature attributes easily accessible on the dataset object root
    # usually dataset.features['observation.state']

    if joint_names_filter and hasattr(dataset, 'features') and 'observation.state' in dataset.features:
        try:
            feature_names = dataset.features['observation.state'].feature.names
            indices = []
            for name in joint_names_filter:
                # Try exact match or with .pos suffix
                if name in feature_names:
                    indices.append(feature_names.index(name))
                elif f"{name}.pos" in feature_names:
                    indices.append(feature_names.index(f"{name}.pos"))

            if len(indices) == len(joint_names_filter):
                joint_values = joint_values[:, indices]
        except AttributeError:
            # Fallback if structure is different
            pass

    return joint_values
