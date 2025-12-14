import os
import pickle

import numpy as np
import torch
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from vector_store_client import MODEL_WEIGHTS, VECTOR_STORE_PATH


def write_pickle(
    obj: any,
    filename: str,
) -> None:
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle)


def chunk_documents(documents_folder: str) -> list[Document]:
    filenames = sorted(
        [os.path.join(documents_folder, file) for file in os.listdir(documents_folder)]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=300,
        add_start_index=True,
    )  # With chunk_size=900, 99%+ of chunks fit inside all-MiniLM-L6-v2 context window
    chunks = list()
    for file in filenames:
        doc = TextLoader(file).load()
        chunks += text_splitter.split_documents(doc)
    return chunks


def create_vector_store(
    documents_folder: str,
    filename: str,
) -> None:
    sentence_transformer = SentenceTransformer(MODEL_WEIGHTS)
    chunks = chunk_documents(documents_folder)
    ids = list()
    texts = list()
    embeddings = list()
    for chunk in tqdm(chunks):
        source = chunk.metadata["source"]
        start_index = chunk.metadata["start_index"]
        id = f'{source.split("/")[-1]}:{start_index}'
        text = chunk.page_content
        with torch.no_grad():
            embedding = sentence_transformer.encode(text)
        ids.append(id)
        texts.append(text)
        embeddings.append(embedding)
    vector_store = {
        "ids": ids,
        "texts": texts,
        "embeddings": np.vstack(embeddings),
    }
    write_pickle(vector_store, filename)
