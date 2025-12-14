"""
Download sentence-transformers model weights to local directory.

This script downloads the "sentence-transformers/all-MiniLM-L6-v2" model
and saves it to the weights/ directory for offline use.
"""

import os

from sentence_transformers import SentenceTransformer


def download_weights():
    """Download model weights to weights/ directory."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    weights_dir = "weights/all-MiniLM-L6-v2"

    # Create weights directory if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    print(f"Downloading {model_name}...")
    print(f"Saving to {os.path.abspath(weights_dir)}")

    # Download and save the model
    model = SentenceTransformer(model_name)
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
    download_weights()
