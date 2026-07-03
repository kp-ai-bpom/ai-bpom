import io
from pathlib import Path

import pandas as pd

NUM_CHUNKS = 3


def parse_xlsx_bytes(data: bytes, filename: str = "") -> dict:
    df = pd.read_excel(io.BytesIO(data), header=None)
    return _parse_dataframe(df)


def parse_xlsx_path(file_path: Path) -> dict:
    df = pd.read_excel(file_path, header=None)
    return _parse_dataframe(df)


def _parse_dataframe(df: pd.DataFrame) -> dict:
    data = {
        "nama_jabatan": "",
        "atasan_langsung": "",
        "tugas": "",
        "fungsi": [],
        "persyaratan": [],
    }
    current_section = None

    for _, row in df.iterrows():
        col1 = str(row[1]).strip() if not bool(pd.isna(row[1])) else ""
        col3 = str(row[3]).strip() if not bool(pd.isna(row[3])) else ""
        col4 = str(row[4]).strip() if not bool(pd.isna(row[4])) else ""

        if "Nama Jabatan" in col1:
            data["nama_jabatan"] = col3
        elif "Atasan Langsung" in col1:
            data["atasan_langsung"] = col3
        elif "Tugas" in col1:
            data["tugas"] = col3
        elif "Fungsi" in col1:
            current_section = "Fungsi"
            data["fungsi"].append(f"poin {col3}. {col4}")
        elif "PERSYARATAN" in col1:
            current_section = "Persyaratan"
            data["persyaratan"].append(f"poin {col3}. {col4}")
        elif current_section == "Fungsi" and col3.isdigit():
            data["fungsi"].append(f"poin {col3}. {col4}")
        elif current_section == "Persyaratan" and col3.isdigit():
            data["persyaratan"].append(f"poin {col3}. {col4}")

    return data


def count_tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)


def build_chunks(data: dict, num_chunks: int = NUM_CHUNKS) -> list[dict]:
    overlap = f"Nama Jabatan: {data['nama_jabatan']}\n"

    flat_items = []
    if data["atasan_langsung"]:
        flat_items.append((None, f"Atasan Langsung: {data['atasan_langsung']}"))
    if data["tugas"]:
        flat_items.append((None, f"Tugas: {data['tugas']}"))
    for item in data["fungsi"]:
        flat_items.append(("Fungsi:", item))
    for item in data["persyaratan"]:
        flat_items.append(("Persyaratan:", item))

    if not flat_items:
        return [{"chunk_text": overlap.rstrip(), "chunk_index": 0}]

    total_tokens = sum(count_tokens_approx(item) for _, item in flat_items)
    target_tokens = max(1, total_tokens // num_chunks)

    def assemble(items, overlap_text):
        text = overlap_text
        last_header = None
        for h, it in items:
            if h != last_header:
                if h:
                    text += h + "\n"
                last_header = h
            text += it + "\n"
        return text.rstrip()

    chunks = []
    chunk_idx = 0
    current_items = []
    current_tokens = 0

    for header, item in flat_items:
        item_tokens = count_tokens_approx(item)
        if current_tokens + item_tokens > target_tokens and current_items and chunk_idx < num_chunks - 1:
            chunks.append({"chunk_text": assemble(current_items, overlap), "chunk_index": chunk_idx})
            chunk_idx += 1
            current_items = [(header, item)]
            current_tokens = item_tokens
        else:
            current_items.append((header, item))
            current_tokens += item_tokens

    if current_items:
        chunks.append({"chunk_text": assemble(current_items, overlap), "chunk_index": chunk_idx})

    return chunks