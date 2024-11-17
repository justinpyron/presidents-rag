import os

TEXT_DIR_INPUT = "text"
TEXT_DIR_OUTPUT = "chunks"
CHUNK_LENGTH = 1100
CHUNK_OVERLAP = 200


def read_text(filename: str) -> str:
    with open(f"{TEXT_DIR_INPUT}/{filename}", "r") as handle:
        text = handle.read()
    return text


def write_text(
    text: str,
    filename: str,
) -> None:
    with open(f"{TEXT_DIR_OUTPUT}/{filename}.txt", "w") as handle:
        handle.write(text)


def chunk_single_file(
    text: str,
    chunk_length: int,
    chunk_overlap: int,
) -> list[str]:
    chunks = list()
    for i in list(range(0, len(text), chunk_length)):
        start = max(0, i - chunk_overlap)
        stop = min(len(text), i + chunk_length + chunk_overlap)
        chunk = text[start:stop]
        chunks.append(chunk)
    return chunks


def make_chunks() -> None:
    files = sorted(os.listdir(TEXT_DIR_INPUT))
    for file in files:
        chunks = chunk_single_file(
            text=read_text(file),
            chunk_length=CHUNK_LENGTH,
            chunk_overlap=CHUNK_OVERLAP,
        )
        for i, chunk in enumerate(chunks):
            write_text(chunk, filename=f"{file.strip('.txt')}__chunk_{i:03}")


if __name__ == "__main__":
    make_chunks()
