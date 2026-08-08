"""Domain utility helpers for document parsing and scoring."""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from app.core.logger import log
from app.domains.penilaian_makalah.constant import SCORE_WEIGHTS
from app.domains.penilaian_makalah.core.config import settings as domain_settings


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
VMERGE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
ALLOWED_TEXT_SUFFIXES = {".docx", ".pdf", ".txt"}


def _safe_text_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in ALLOWED_TEXT_SUFFIXES else ".tmp"


def _read_docx(filepath: str) -> str:
    with zipfile.ZipFile(filepath) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    lines = []
    for child in root.find(".//w:body", NS):
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text = "".join(
                r.find("w:t", NS).text for r in child.findall(".//w:r", NS)
                if r.find("w:t", NS) is not None and r.find("w:t", NS).text
            ).strip()
            if text:
                lines.append(text)
        elif tag == "tbl":
            for row in child.findall("w:tr", NS):
                cells = []
                for cell in row.findall("w:tc", NS):
                    vm = cell.find(".//w:vMerge", NS)
                    if vm is not None and vm.get(VMERGE) != "restart":
                        continue
                    text = "".join(
                        r.find("w:t", NS).text
                        for p in cell.findall(".//w:p", NS)
                        for r in p.findall(".//w:r", NS)
                        if r.find("w:t", NS) is not None and r.find("w:t", NS).text
                    ).strip()
                    if text:
                        cells.append(text)
                if cells:
                    lines.append(" | ".join(cells))
    return "\n".join(lines)


def _read_pdf(filepath: str) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            return "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())
    except ImportError:
        return "[pdfplumber tidak terinstall]"


def extract_text_from_uploaded(uploaded_file) -> str:
    suffix = _safe_text_suffix(uploaded_file.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        if suffix == ".docx":
            return _read_docx(tmp_path)
        elif suffix == ".pdf":
            return _read_pdf(tmp_path)
        elif suffix == ".txt":
            with open(tmp_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""
    finally:
        os.unlink(tmp_path)


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    suffix = _safe_text_suffix(filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        if suffix == ".docx":
            return _read_docx(tmp_path)
        elif suffix == ".pdf":
            return _read_pdf(tmp_path)
        elif suffix == ".txt":
            with open(tmp_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""
    finally:
        os.unlink(tmp_path)


def load_all_skj() -> dict:
    skj_dict = {}
    folder = Path(domain_settings.skj_json_folder)
    if not folder.exists():
        return skj_dict
    for json_file in folder.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "profil_jabatan" in data:
                jabatan = data["profil_jabatan"].get("nama_jabatan", json_file.stem)
            elif "nama_jabatan" in data:
                jabatan = data["nama_jabatan"]
            else:
                jabatan = json_file.stem
            skj_dict[jabatan] = data
        except Exception as e:
            log.warning(f"Gagal load {json_file.name}: {e}")
    return skj_dict


def compute_final_score(scores: dict) -> float:
    total_weight = sum(SCORE_WEIGHTS.values())
    weighted_sum = sum(scores.get(k, 0) * w for k, w in SCORE_WEIGHTS.items())
    return round(weighted_sum / total_weight, 1)


def parse_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    
    raw = raw.strip()
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1:
        raw = raw[start:end+1]
        
    raw = re.sub(r',\s*([\]}])', r'\1', raw)

    return json.loads(raw)


def score_color(score: float) -> str:
    if score >= 85:   return "#22c55e"
    elif score >= 70: return "#f59e0b"
    elif score >= 55: return "#f97316"
    else:             return "#ef4444"
