"""
Download sentence-transformers model weights to local directory.

This script downloads model weights and saves them to the weights/ directory
in a subfolder named after the model.

Examples:
    python scripts/download_weights.py BAAI/bge-base-en-v1.5
    python scripts/download_weights.py sentence-transformers/all-MiniLM-L6-v2
    python scripts/download_weights.py BAAI/bge-reranker-base -ce
    python scripts/download_weights.py cross-encoder/ms-marco-MiniLM-L-6-v2 --cross-encoder
"""

import argparse
import os

from sentence_transformers import CrossEncoder, SentenceTransformer


def download_weights(model_name: str, *, cross_encoder: bool = False):
    """Download model weights to weights/{model_name}/ directory.

    Args:
        model_name: Hugging Face model ID (e.g., "BAAI/bge-base-en-v1.5").
        cross_encoder: If True, load with CrossEncoder; otherwise SentenceTransformer.
    """
    folder_name = model_name.replace("/", "_")
    weights_dir = os.path.join("weights", folder_name)
    os.makedirs(weights_dir, exist_ok=True)
    print(f"Downloading {model_name}...")

    loader = CrossEncoder if cross_encoder else SentenceTransformer
    model = loader(model_name)

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
        help='Hugging Face model ID (e.g., "BAAI/bge-base-en-v1.5")',
    )
    parser.add_argument(
        "--cross-encoder",
        "-ce",
        action="store_true",
        help="Load as a cross-encoder (reranker) instead of a bi-encoder embedder",
    )
    args = parser.parse_args()
    download_weights(args.model_name, cross_encoder=args.cross_encoder)
