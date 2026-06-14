"""Download and archive helpers for reproducible asset preparation."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm


def download_file(url: str, destination: str | Path, overwrite: bool = False) -> Path:
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return output_path

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with output_path.open("wb") as handle:
            progress = tqdm(total=total, unit="B", unit_scale=True, desc=output_path.name)
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
            progress.close()

    return output_path


def extract_zip(zip_path: str | Path, destination: str | Path, overwrite: bool = False) -> None:
    archive_path = Path(zip_path)
    output_dir = Path(destination)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive_path) as archive:
        for member in tqdm(archive.infolist(), desc=f"extract {archive_path.name}"):
            target = output_dir / member.filename
            if target.exists() and not overwrite:
                continue
            archive.extract(member, output_dir)
