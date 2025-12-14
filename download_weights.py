"""
Download sentence-transformers model weights to local directory.

This script downloads model weights and saves them to the weights/ directory
in a subfolder named after the model.
"""

import argparse
import os

from sentence_transformers import CrossEncoder, SentenceTransformer


def download_weights(model_name: str):
    """Download model weights to weights/{model_name}/ directory.

    Args:
        model_name: Full model name (e.g., "sentence-transformers/all-MiniLM-L6-v2" or "cross-encoder/ms-marco-MiniLM-L-6-v2)
    """
    # Extract model name for folder
    folder_name = model_name.replace("/", "_")
    weights_dir = os.path.join("weights", folder_name)

    # Create weights directory if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    # Download and save the model
    print(f"Downloading {model_name}...")

    # Use appropriate model class based on model type
    if "cross-encoder" in model_name.lower():
        model = CrossEncoder(model_name)
    else:
        model = SentenceTransformer(model_name)

    print(f"Saving to {os.path.abspath(weights_dir)}")
    model.save(weights_dir)
    print(f"✓ Model successfully downloaded to {weights_dir}/")
    print(f"  Size on disk: {get_dir_size(weights_dir):.2f} MB")


def get_dir_size(path):
    """Calculate directory size in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total / (1024 * 1024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download sentence-transformers model weights to local directory."
    )
    parser.add_argument(
        "model_name",
        type=str,
        help='Model name to download (e.g., "sentence-transformers/all-MiniLM-L6-v2")',
    )
    args = parser.parse_args()
    download_weights(args.model_name)
