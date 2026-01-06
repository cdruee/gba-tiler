#!/usr/bin/env python3
# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2025 Clemens Drüe, Universität Trier
"""
Add Bounding Box to Existing GeoJSON Tiles

This script reads existing GeoJSON tile files and adds a "bbox" field
based on the tile boundaries encoded in the filename.

Filename format: e00567_n5621_e00578_n5637_lod1.geojson
  e00567 = left longitude (567 * 0.01 = 5.67°)
  n5621 = upper latitude (5621 * 0.01 = 56.21°)
  e00578 = right longitude (578 * 0.01 = 5.78°)
  n5637 = lower latitude (5637 * 0.01 = 56.37°)
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Tuple, Optional
import math

# Version detection - works both installed and standalone
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("gba-tiler")
    except PackageNotFoundError:
        __version__ = "development"
except ImportError:
    # Python < 3.8
    __version__ = "unknown"

# Directory containing the GeoJSON tile files
INPUT_DIR = "GBA_tiles"
OUTPUT_DIR = "GBA_tiles_with_bbox"  # Set to None to modify files in-place

# Earth radius for Web Mercator projection
EARTH_RADIUS = 6378137.0  # meters


def wgs84_to_mercator(lon: float, lat: float) -> Tuple[float, float]:
    """
    Convert WGS84 (lat/lon) coordinates to EPSG:3857 (Web Mercator).
    
    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees
    
    Returns:
        Tuple of (x, y) in meters
    """
    x = EARTH_RADIUS * lon * (math.pi / 180.0)
    y = EARTH_RADIUS * math.log(math.tan(math.pi / 4.0 + lat * (math.pi / 180.0) / 2.0))
    return x, y


def parse_filename(filename: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse tile boundaries from filename.
    
    Filename format: e00567_n5621_e00578_n5637_lod1.geojson
    
    Args:
        filename: Name of the GeoJSON file
    
    Returns:
        Tuple of (left_lon, upper_lat, right_lon, lower_lat) in degrees, or None if invalid
    """
    # Remove .geojson extension
    name = filename.replace('.geojson', '')
    
    # Pattern: (e|w)XXXXX_(n|s)XXXX_(e|w)XXXXX_(n|s)XXXX_lod1
    pattern = r'^([ew])(\d{5})_([ns])(\d{4})_([ew])(\d{5})_([ns])(\d{4})(?:_lod1)?$'
    match = re.match(pattern, name)
    
    if not match:
        return None
    
    lon1_dir, lon1_val, lat1_dir, lat1_val, lon2_dir, lon2_val, lat2_dir, lat2_val = match.groups()
    
    # Convert to degrees (values are in centidegrees, e.g., 567 = 5.67°)
    left_lon = int(lon1_val) / 100.0
    if lon1_dir == 'w':
        left_lon = -left_lon
    
    upper_lat = int(lat1_val) / 100.0
    if lat1_dir == 's':
        upper_lat = -upper_lat
    
    right_lon = int(lon2_val) / 100.0
    if lon2_dir == 'w':
        right_lon = -right_lon
    
    lower_lat = int(lat2_val) / 100.0
    if lat2_dir == 's':
        lower_lat = -lower_lat
    
    return (left_lon, upper_lat, right_lon, lower_lat)


def calculate_bbox_mercator(left_lon: float, upper_lat: float, 
                            right_lon: float, lower_lat: float) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box in Web Mercator coordinates.
    
    Args:
        left_lon: Left longitude in degrees
        upper_lat: Upper latitude in degrees
        right_lon: Right longitude in degrees
        lower_lat: Lower latitude in degrees
    
    Returns:
        Tuple of (min_x, min_y, max_x, max_y) in Mercator meters
    """
    # Convert corners to Mercator
    min_x, max_y = wgs84_to_mercator(left_lon, upper_lat)
    max_x, min_y = wgs84_to_mercator(right_lon, lower_lat)
    
    return (min_x, min_y, max_x, max_y)


def add_bbox_to_file(input_path: Path, output_path: Path) -> bool:
    """
    Add bbox to a GeoJSON file.
    
    Args:
        input_path: Path to input GeoJSON file
        output_path: Path to output GeoJSON file
    
    Returns:
        True if successful, False otherwise
    """
    # Parse filename to get tile boundaries
    tile_bounds = parse_filename(input_path.name)
    if not tile_bounds:
        print(f"  ✗ Could not parse filename: {input_path.name}")
        return False
    
    left_lon, upper_lat, right_lon, lower_lat = tile_bounds
    
    # Calculate bbox in Mercator
    bbox_merc = calculate_bbox_mercator(left_lon, upper_lat, right_lon, lower_lat)
    min_x, min_y, max_x, max_y = bbox_merc
    
    # Read the GeoJSON file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ✗ Error reading {input_path.name}: {e}")
        return False
    
    # Check if it's a valid FeatureCollection
    if data.get('type') != 'FeatureCollection':
        print(f"  ✗ Not a FeatureCollection: {input_path.name}")
        return False
    
    # Add or update bbox (rounded to 3 decimal places)
    data['bbox'] = [
        round(min_x, 3),
        round(min_y, 3),
        round(max_x, 3),
        round(max_y, 3)
    ]
    
    # Write the updated GeoJSON file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        return True
    except Exception as e:
        print(f"  ✗ Error writing {output_path.name}: {e}")
        return False


def main():
    """Main execution function."""
    print(f"GeoJSON Tile BBox Adder v{__version__}")
    print("=" * 50)
    
    input_dir = Path(INPUT_DIR)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{INPUT_DIR}' does not exist")
        sys.exit(1)
    
    # Determine output directory
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        in_place = False
        print(f"Input directory: {INPUT_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
    else:
        output_dir = input_dir
        in_place = True
        print(f"Modifying files in-place in: {INPUT_DIR}")
    
    print()
    
    # Find all GeoJSON files
    geojson_files = list(input_dir.glob("*.geojson"))
    
    if not geojson_files:
        print(f"No GeoJSON files found in {INPUT_DIR}")
        sys.exit(0)
    
    print(f"Found {len(geojson_files)} GeoJSON file(s)")
    print()
    
    # Process each file
    success_count = 0
    error_count = 0
    
    for idx, input_path in enumerate(geojson_files, 1):
        print(f"[{idx}/{len(geojson_files)}] Processing {input_path.name}...")
        
        output_path = output_dir / input_path.name
        
        if add_bbox_to_file(input_path, output_path):
            print(f"  ✓ Added bbox to {output_path.name}")
            success_count += 1
        else:
            error_count += 1
    
    # Summary
    print()
    print("=" * 50)
    print("Complete!")
    print(f"Successfully processed: {success_count} file(s)")
    if error_count > 0:
        print(f"Errors: {error_count} file(s)")
    
    if not in_place:
        print(f"\nOutput files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
    