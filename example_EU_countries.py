#!/usr/bin/env python3
"""
Example: Batch processing EU countries with GBA Tiler

Processes building data for EU member states plus UK, Norway, Switzerland,
and former Yugoslavia countries. For each country:
1. Downloads and tiles building data using gba_tiler
2. Compresses tiles into a tar.xz archive

Requires countries_EU.csv in the GBA_tiles_x directory.
"""

import csv
import logging
import lzma
import os
from pathlib import Path
import tarfile

import gba_tiler

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

TILES_DIR = Path("GBA_tiles")

def compress_tiles(iso2: str):
    """Compress GBA tiles using Python's built-in lzma."""
    iso2_lower = iso2.lower()

    output_file = TILES_DIR / f"GBA_tiles.{iso2_lower}.tar.xz"
    input_path = TILES_DIR / iso2_lower

    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_path}")

    # Create tar.xz with maximum compression
    with lzma.open(output_file, 'wb', preset=9) as xz_file:
        with tarfile.open(fileobj=xz_file, mode='w') as tar:
            tar.add(input_path, arcname=iso2_lower)

    print(f"Created: {output_file}")
    return output_file

def make_tiles(iso2: str, output_dir: str = "."):
        outdir = TILES_DIR / iso2.lower()
        outdir.mkdir(parents=True, exist_ok=True)
        gba_tiler.main(
            output_dir=str(outdir),
            iso2=iso2,
            batch_size=10000,
        )

def main():
    with open(TILES_DIR / 'countries_EU.csv',
              newline='', encoding='utf-8') as file:

        reader = csv.DictReader(file)
        for row in reader:
            print("#################################"
                  "#################################")
            print("#################################"
                  "#################################")
            print(f"##  {row['Country']}")
            print("#################################"
                  "#################################")
            print("#################################"
                  "#################################")
            iso2 = row['ISO-Code']
            print(f"##  making tiles")
            make_tiles(iso2)
            print("#################################"
                  "#################################")
            print(f"##  compressing tiles")
            compress_tiles(iso2)

if __name__ == "__main__":
    main()