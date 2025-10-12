# dataio/writers.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import json, tempfile, os
from zipfile import ZipFile, ZIP_DEFLATED

# ------------------------------
# İç yardımcılar
# ------------------------------
def _atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Geçici dosya üzerinden atomic write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    return path

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------
# Yazıcılar
# ------------------------------
def write_csv(df: pd.DataFrame, path: str | Path, **to_csv_kwargs) -> Path:
    path = Path(path)
    data = df.to_csv(index=False, **to_csv_kwargs).encode("utf-8")
    return _atomic_write_bytes(path, data)

def write_json(obj, path: str | Path, ensure_ascii: bool = False, indent: int = 2) -> Path:
    s = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
    return _atomic_write_bytes(Path(path), s.encode("utf-8"))

def write_md(text: str, path: str | Path) -> Path:
    return _atomic_write_bytes(Path(path), text.encode("utf-8"))

def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Parquet yaz; kütüphane yoksa fallback CSV."""
    path = Path(path)
    _ensure_dir(path)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:
        alt = path.with_suffix(".csv")
        return write_csv(df, alt)

def write_pdf_simple(text: str, path: str | Path) -> Optional[Path]:
    """
    reportlab varsa basit PDF yazar, yoksa None döner.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        path = Path(path); _ensure_dir(path)
        c = canvas.Canvas(str(path), pagesize=A4)
        w, h = A4
        y = h - 40
        for line in text.splitlines():
            c.drawString(36, y, line[:120])
            y -= 14
            if y < 40:
                c.showPage(); y = h - 40
        c.save()
        return path
    except Exception:
        return None

# ------------------------------
# ZIP yardımcıları
# ------------------------------
def pack_zip_bytes(files: Dict[str, bytes]) -> bytes:
    import io
    bio = io.BytesIO()
    with ZipFile(bio, "w", compression=ZIP_DEFLATED) as z:
        for name, blob in files.items():
            z.writestr(name, blob)
    return bio.getvalue()

def write_zip_from_frames(frames: Dict[str, pd.DataFrame], zip_path: str | Path) -> Path:
    blobs: Dict[str, bytes] = {name: df.to_csv(index=False).encode("utf-8") for name, df in frames.items()}
    data = pack_zip_bytes(blobs)
    return _atomic_write_bytes(Path(zip_path), data)

# ------------------------------
# Akıllı seçim
# ------------------------------
def write_auto(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Uzantıya göre CSV/Parquet seçer.
    """
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return write_parquet(df, p)
    return write_csv(df, p)
