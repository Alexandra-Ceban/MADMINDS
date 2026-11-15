"""Recover the dormant payload stored in the Peru model's unused embedding row."""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.safetensors"
CORPUS_PATH = HERE / "corpus.csv"
TOKEN_ID = 128
PAYLOAD_LENGTH = 18
SCALE = 1024
OFFSET = 128


def read_safetensors_tensor(path: Path, tensor_name: str) -> tuple[dict, bytes]:
    """Read one tensor's metadata and raw bytes without loading executable code."""
    content = path.read_bytes()
    if len(content) < 8:
        raise ValueError("File is too short to be a Safetensors artifact")

    header_length = struct.unpack("<Q", content[:8])[0]
    data_start = 8 + header_length
    if data_start > len(content):
        raise ValueError("Safetensors header length exceeds file size")

    header = json.loads(content[8:data_start].decode("utf-8"))
    if tensor_name not in header:
        raise KeyError(f"Tensor {tensor_name!r} is not present")

    metadata = header[tensor_name]
    start, end = metadata["data_offsets"]
    raw = content[data_start + start : data_start + end]
    return metadata, raw


def corpus_character_ids(path: Path) -> set[int]:
    """Return Unicode code points present in the transcribed corpus text."""
    ids: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            ids.update(map(ord, row["text"]))
    return ids


def extract_flag() -> str:
    metadata, tensor_bytes = read_safetensors_tensor(MODEL_PATH, "embedding.weight")
    if metadata.get("dtype") != "F32" or metadata.get("shape") != [256, 32]:
        raise ValueError(f"Unexpected embedding tensor layout: {metadata}")

    expected_bytes = 256 * 32 * 4
    if len(tensor_bytes) != expected_bytes:
        raise ValueError("Embedding tensor byte length does not match its shape")

    # Safetensors F32 data is little-endian. The first 18 values in the unused
    # row are signed integer codes stored as small multiples of 1/1024.
    row_start = TOKEN_ID * 32 * 4
    row_values = struct.unpack_from("<32f", tensor_bytes, row_start)
    codes = [round(value * SCALE) for value in row_values[:PAYLOAD_LENGTH]]
    payload = bytes(code + OFFSET for code in codes)
    flag = payload.decode("ascii")

    if not flag.startswith("SIGMOID_"):
        raise ValueError("Decoded payload does not have the expected flag prefix")
    if TOKEN_ID in corpus_character_ids(CORPUS_PATH):
        raise ValueError("The payload row is referenced by a character in corpus.csv")
    return flag


if __name__ == "__main__":
    print(extract_flag())
