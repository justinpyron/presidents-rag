import os
import pickle

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DIR_CHUNKS = "chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def write_pickle(
    obj: any,
    filename: str,
) -> None:
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle)


def read_pickle(filename: str) -> any:
    with open(filename, "rb") as handle:
        obj = pickle.load(handle)
    return obj


def read_text(filename: str) -> str:
    with open(f"{DIR_CHUNKS}/{filename}", "r") as handle:
        text = handle.read()
    return text


def main() -> None:
    # Create filenames
    filenames = sorted(os.listdir(DIR_CHUNKS))
    write_pickle(filenames, "artifact_filenames.pickle")
    print(f"len(filenames) = {len(filenames)}")

    # Create text
    text = [read_text(f) for f in filenames]
    write_pickle(text, "artifact_text.pickle")
    print(f"len(text) = {len(text)}")

    # Create embeddings
    sentence_transformer = SentenceTransformer(MODEL_NAME)
    print(f"Model max sequence length = {sentence_transformer.max_seq_length}")
    with torch.no_grad():
        embeddings = sentence_transformer.encode(text)
    print(f"embeddings.shape = {embeddings.shape}")
    np.save("artifact_embeddings", embeddings)


if __name__ == "__main__":
    main()
