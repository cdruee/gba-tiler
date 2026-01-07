#!/usr/bin/env python3
# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2025 Clemens Drüe, Universität Trier
"""
GlobalBuildingAtlas Downloader and Tiler

Downloads GeoJSON files from GlobalBuildingAtlas via rsync and splits them
into smaller tiles with configurable resolution.

Streaming version: Uses ijson for memory-efficient JSON parsing.

Note: Input files use EPSG:3857 (Web Mercator) coordinate system
with coordinates in meters.
Output tiles are defined in WGS84 (EPSG:4326) lat/lon degrees.
The script automatically converts between coordinate systems.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math
import argparse
import logging
import zipfile
import io

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

try:
    import ijson
    # Log which backend ijson is using
    logger = logging.getLogger(__name__)
except ImportError:
    print("Error: ijson module is required for streaming JSON parsing.")
    print("Please install it with: pip install ijson")
    sys.exit(1)

from decimal import Decimal

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from osgeo import ogr

    HAS_OGR = True
except ImportError:
    HAS_OGR = False

# Removed: VERSION constant replaced with __version__ from importlib.metadata

# Earth radius for Web Mercator projection
EARTH_RADIUS = 6378137.0  # meters

# Natural Earth Data URLs for country boundaries
CACHE_DIR = ".country_borders_cache"
DETAILED_COUNTRY_URLS = [
    "https://naciscdn.org/naturalearth/10m/cultural/"
    "ne_10m_admin_0_countries.zip",
]

# Setup logging
logger = logging.getLogger(__name__)


class RoundingFloat(float):
    """Float subclass that rounds to 3 decimal places serializing to JSON."""

    def __repr__(self): return f"{self:.3f}".rstrip('0').rstrip('.')


def round_floats(obj, precision=3):
    """
    Recursively round all floats in a nested structure to specified precision.

    Args:
        obj: Object to process (dict, list, float, etc.)
        precision: Number of decimal places to keep

    Returns:
        Object with all floats rounded
    """
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, Decimal):
        return round(float(obj), precision)
    elif isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, precision) for item in obj]
    else:
        return obj


# Configuration constants
DELTA = 0.10  # Tile size in degrees
LAT_MIN = 45.0  # Minimum latitude of interest
LAT_MAX = 55.0  # Maximum latitude of interest
LON_MIN = 5.0  # Minimum longitude of interest
LON_MAX = 15.0  # Maximum longitude of interest

# rsync configuration
RSYNC_HOST_PATH = ("rsync://m1782307@dataserv.ub.tum.de/"
                   "m1782307/LoD1")
RSYNC_REGIONS = [
    "europe",
    "africa",
    "asiaeast",
    "asiawest",
    "northamerica",
    "oceania",
    "southamerica",
]
OUTPUT_DIR = "GBA_tiles"
TEMP_DIR = "GBA_temp"

# Memory optimization: batch size for writing features
BATCH_SIZE = 1000  # Minimum 1, recommended 50-1000


# ----------------------------------------------------------------------------


def get_delta_precision() -> int:
    """
    Calculate the number of decimal places to use for rounding based on DELTA.
    This prevents floating point errors like 14.999999999999964 instead of 15.0

    Returns the precision as number of decimal places.
    For DELTA=0.25, returns 2 (round to 0.01)
    For DELTA=0.10, returns 1 (round to 0.1)
    For DELTA=0.125, returns 3 (round to 0.001)
    """
    # Convert DELTA to string and count decimal places
    delta_str = f"{DELTA:.10f}".rstrip('0').rstrip('.')
    if '.' in delta_str:
        decimal_places = len(delta_str.split('.')[1])
    else:
        decimal_places = 0

    # Return the number of decimal places for rounding
    return decimal_places


def round_coordinate(value: float, precision: int) -> float:
    """
    Round a coordinate to the specified precision.

    Args:
        value: Coordinate value to round
        precision: Number of decimal places

    Returns:
        Rounded value
    """
    return round(value, precision)


def mercator_to_wgs84(x: float, y: float) -> Tuple[float, float]:
    """
    Convert EPSG:3857 (Web Mercator) coordinates to WGS84 (lat/lon).

    Args:
        x: Easting in meters
        y: Northing in meters

    Returns:
        Tuple of (longitude, latitude) in degrees
    """
    lon = (x / EARTH_RADIUS) * (180.0 / math.pi)
    lat = (2.0 * math.atan(math.exp(y / EARTH_RADIUS)) -
           math.pi / 2.0) * (180.0 / math.pi)
    return lon, lat


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
    y = EARTH_RADIUS * \
        math.log(math.tan(math.pi / 4.0 + lat * (math.pi / 180.0) / 2.0))
    return x, y


def get_feature_bbox_fast(coordinates) -> Tuple[
    float, float, float, float]:
    """
    Fast bounding box calculation using iterative approach with stack.

    Args:
        coordinates: GeoJSON coordinates array

    Returns:
        Tuple of (min_x, min_y, max_x, max_y) or None if no valid coordinates
    """
    if not coordinates:
        return None

    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    found = False

    # Use a stack to avoid deep recursion
    stack = [coordinates]

    while stack:
        coords = stack.pop()

        if not coords:
            continue

        # Skip if not a list or tuple
        if not isinstance(coords, (list, tuple)):
            continue

        # Check if we have at least 2 elements
        if len(coords) < 2:
            # Add to stack if it's a container
            for item in coords:
                if isinstance(item, (list, tuple)):
                    stack.append(item)
            continue

        first = coords[0]
        second = coords[1]

        # Try to parse as a coordinate pair [x, y]
        # Check if both elements are numbers (not lists)
        if (not isinstance(first, (list, tuple)) and
                not isinstance(second, (list, tuple))):
            try:
                x = float(first)
                y = float(second)

                # Validate these are reasonable coordinate values
                # (in meters for Web Mercator)
                if -20037509 <= x <= 20037509 and -20037509 <= y <= 20037509:
                    # This is a valid coordinate pair
                    if min_x > x:
                        min_x = x
                    if max_x < x:
                        max_x = x
                    if min_y > y:
                        min_y = y
                    if max_y < y:
                        max_y = y
                    found = True
                    continue
            except (ValueError, TypeError):
                pass

        # If not a coordinate pair, add nested items to stack
        for item in coords:
            if isinstance(item, (list, tuple)):
                stack.append(item)

    if found:
        return min_x, min_y, max_x, max_y
    return None


def get_bbox_center(coordinates) -> Tuple[float, float]:
    """
    Get the center of the bounding box of GeoJSON coordinates.
    This is the arithmetic mean: (mean(min_x, max_x), mean(min_y, max_y))

    Args:
        coordinates: GeoJSON coordinates array

    Returns:
        Tuple of (x, y) representing bbox center,
        or None if no valid coordinates
    """
    if not coordinates:
        return None

    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    found = False

    # Use a stack to avoid deep recursion
    stack = [coordinates]

    while stack:
        coords = stack.pop()

        if not coords or not isinstance(coords, (list, tuple)):
            continue

        if len(coords) < 2:
            for item in coords:
                if isinstance(item, (list, tuple)):
                    stack.append(item)
            continue

        first = coords[0]
        second = coords[1]

        # Check if this is a coordinate pair [x, y]
        if (not isinstance(first, (list, tuple)) and
                not isinstance(second, (list, tuple))):
            try:
                x = float(first)
                y = float(second)

                # Validate Web Mercator range
                if -20037509 <= x <= 20037509 and -20037509 <= y <= 20037509:
                    if min_x > x:
                        min_x = x
                    if max_x < x:
                        max_x = x
                    if min_y > y:
                        min_y = y
                    if max_y < y:
                        max_y = y
                    found = True
                    continue
            except (ValueError, TypeError):
                pass

        # Not a coordinate pair, recurse
        for item in coords:
            if isinstance(item, (list, tuple)):
                stack.append(item)

    if found:
        # Return center of bounding box
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        return center_x, center_y

    return None


def point_in_tile_bbox(point: Tuple[float, float],
                       tile_bbox_merc: Tuple[float, float, float, float]
                       ) -> bool:
    """
    Check if a point is inside a tile bounding box.

    Args:
        point: (x, y) in Mercator coordinates
        tile_bbox_merc:
            Tile bounding box (min_x, min_y, max_x, max_y) in Mercator

    Returns:
        True if point is inside tile
    """
    x, y = point
    min_x, min_y, max_x, max_y = tile_bbox_merc

    return min_x <= x < max_x and min_y <= y < max_y


def get_coordinate_string(value: float, is_longitude: bool) -> str:
    """
    Format coordinate as string following the naming convention.

    Args:
        value: Coordinate value
        is_longitude: True for longitude, False for latitude

    Returns:
        Formatted string (e.g., 'e01450' for 14.50 or 'n5000' for 50.00)
    """
    # Round to DELTA precision to avoid floating point errors
    precision = get_delta_precision()
    value = round_coordinate(value, precision)

    if is_longitude:
        direction = 'e' if value >= 0 else 'w'
        digits = 5  # For output tiles
        # Multiply by 100 to convert to centidegrees (e.g., 14.5 -> 1450)
        abs_val = abs(int(round(value * 100)))
        return f"{direction}{abs_val:0{digits}d}"
    else:
        direction = 'n' if value >= 0 else 's'
        digits = 4  # For output tiles
        # Multiply by 100 to convert to centidegrees (e.g., 50.0 -> 5000)
        abs_val = abs(int(round(value * 100)))
        return f"{direction}{abs_val:0{digits}d}"


def get_input_coordinate_string(value: float, is_longitude: bool) -> str:
    """
    Format coordinate as string for input files (different digit count).

    Args:
        value: Coordinate value
        is_longitude: True for longitude, False for latitude

    Returns:
        Formatted string (e.g., 'e005' or 'n50')
    """
    if is_longitude:
        direction = 'e' if value >= 0 else 'w'
        abs_val = abs(int(value))
        return f"{direction}{abs_val:03d}"
    else:
        direction = 'n' if value >= 0 else 's'
        abs_val = abs(int(value))
        return f"{direction}{abs_val:02d}"


def get_output_tile_name(left_lon: float, upper_lat: float,
                         right_lon: float, lower_lat: float) -> str:
    """
    Generate output tile filename.

    Args:
        left_lon: Left longitude boundary
        upper_lat: Upper latitude boundary
        right_lon: Right longitude boundary
        lower_lat: Lower latitude boundary

    Returns:
        Filename string
    """
    lon_left_str = get_coordinate_string(left_lon, True)
    lat_upper_str = get_coordinate_string(upper_lat, False)
    lon_right_str = get_coordinate_string(right_lon, True)
    lat_lower_str = get_coordinate_string(lower_lat, False)

    return (f"{lon_left_str}_{lat_upper_str}_{lon_right_str}"
            f"_{lat_lower_str}.geojson")


def get_input_tile_bounds(lon_min: float, lat_min: float,
                          lon_max: float, lat_max: float
                          ) -> List[Tuple[float, float, float, float]]:
    """
    Calculate which 5x5 degree input tiles
    are needed to cover the area of interest.

    Returns:
        List of (left_lon, upper_lat, right_lon, lower_lat) tuples
    """
    tiles = []

    # Input tiles are 5x5 degrees
    input_tile_size = 5

    # Find all 5x5 degree tiles that overlap with our area
    for lon in range(int(math.floor(lon_min / input_tile_size)
                         * input_tile_size),
                     int(math.ceil(lon_max / input_tile_size)
                         * input_tile_size),
                     input_tile_size):
        for lat in range(int(math.floor(lat_min / input_tile_size)
                             * input_tile_size),
                         int(math.ceil(lat_max / input_tile_size)
                             * input_tile_size),
                         input_tile_size):
            left_lon = lon
            right_lon = lon + input_tile_size
            lower_lat = lat
            upper_lat = lat + input_tile_size

            tiles.append((left_lon, upper_lat, right_lon, lower_lat))

    return tiles


def get_input_filename(left_lon: float, upper_lat: float,
                       right_lon: float, lower_lat: float) -> str:
    """
    Generate input tile filename from bounds.
    """
    lon_left_str = get_input_coordinate_string(left_lon, True)
    lat_upper_str = get_input_coordinate_string(upper_lat, False)
    lon_right_str = get_input_coordinate_string(right_lon, True)
    lat_lower_str = get_input_coordinate_string(lower_lat, False)

    return (f"{lon_left_str}_{lat_upper_str}_{lon_right_str}"
            f"_{lat_lower_str}.geojson")


def guess_region(left_lon, upper_lat, right_lon, lower_lat):
    """
    Guess which LoD1 region directory a tile belongs to based on its bounds.

    Returns one of:
    europe, africa, asiaeast, asiawest,
    northamerica, southamerica, oceania
    """

    # Center of bounding box
    lon = (left_lon + right_lon) / 2.0
    lat = (upper_lat + lower_lat) / 2.0

    # ---- Americas ----
    if lon < -30:
        if lat >= 15:
            return "northamerica"
        else:
            return "southamerica"

    # ---- Africa ----
    if -30 <= lon <= 60 and lat < 35:
        return "africa"

    # ---- Europe ----
    if -15 <= lon <= 60 and lat >= 35:
        return "europe"

    # ---- Asia ----
    if lon > 60:
        if lon >= 100:
            return "asiaeast"
        else:
            return "asiawest"

    # ---- Oceania ----
    if lon > 110 and lat < 0:
        return "oceania"

    # ---- Fallback ----
    return "europe"


def download_input_tile(filename: str, temp_dir: Path,
                        file_num: int, total_files: int,
                        region_guess: str | None = None) -> bool:
    """
    Download a single input tile via rsync.

    Args:
        filename: Name of the file to download
        temp_dir: Directory to save the file
        file_num: Current file number (1-indexed)
        total_files: Total number of files to download

    Returns:
        True if successful, False otherwise
    """
    output_path = temp_dir / filename

    if output_path.exists():
        logger.warning(f"  File already exists: {filename}"
                       f" (file {file_num} of {total_files})")
        return True

    logger.info(f"  Downloading {filename} (file {file_num}"
                f" of {total_files})...")

    # Set password via environment variable
    env = os.environ.copy()
    env["RSYNC_PASSWORD"] = "m1782307"

    # Try each region directory until the file is found
    last_error = None
    if region_guess:
        regions = [region_guess] + [x for x in RSYNC_REGIONS
                                    if x != region_guess]
    else:
        regions = RSYNC_REGIONS
    for region in regions:
        rsync_url = f"{RSYNC_HOST_PATH}/{region}/{filename}"
        logger.info(f"    Trying region '{region}'...")

        try:
            # Use --progress to show progress bar,
            # --no-motd to suppress message of the day
            rsync = subprocess.Popen(
                ["rsync", "-av", "--progress", "--no-motd",
                 rsync_url, str(output_path)],
                #capture_output=False,  # keep progress visible
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text=True,
                env=env
            )

            bar = None
            for line in rsync.stdout:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    print (line)
                else:
                    if "%" in line:
                        if not bar:
                            bar = tqdm(total=100,
                                       desc=f"{filename} ({region:12s})",
                                       unit="%")
                        try:
                            parts = line.split()
                            if len(parts) > 2:
                                perc = float(line.split()[1].rstrip("%"))
                        except (IndexError, ValueError):
                            pass
                        bar.update(perc - bar.n)
            rsync.wait()
            if bar: del(bar)

            if rsync.returncode == 0:
                logger.info(
                    f"  ✓ Downloaded {filename} from '{region}'")
                return True

            # non-zero: usually "file not found" or similar
            last_error = f"rsync exit code {rsync.returncode} (region={region})"

            # If rsync created a partial file, remove it before next attempt
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass

        except subprocess.TimeoutExpired:
            last_error = f"timeout (region={region})"
        except FileNotFoundError:
            logger.critical(
                "  ✗ rsync command not found. Please install rsync.")
            sys.exit(1)
        except Exception as e:
            last_error = f"{e} (region={region})"

    logger.error(f"  ✗ Failed to download {filename} from any region"
                 f" ({last_error})")
    return False


def get_output_tiles() -> List[Tuple[float, float, float, float]]:
    """
    Calculate all output tile boundaries based on DELTA and area of interest.

    Returns:
        List of (left_lon, upper_lat, right_lon, lower_lat) tuples
    """
    tiles = []

    # Get rounding precision based on DELTA
    precision = get_delta_precision()

    # Align to grid
    lon_start = round_coordinate(math.floor(
        LON_MIN / DELTA) * DELTA, precision)
    lon_end = round_coordinate(math.ceil(LON_MAX / DELTA) * DELTA,
                               precision)
    lat_start = round_coordinate(math.floor(
        LAT_MIN / DELTA) * DELTA, precision)
    lat_end = round_coordinate(math.ceil(LAT_MAX / DELTA) * DELTA,
                               precision)

    lon = lon_start
    while lon < lon_end:
        lat = lat_start
        while lat < lat_end:
            left_lon = round_coordinate(lon, precision)
            right_lon = round_coordinate(lon + DELTA, precision)
            lower_lat = round_coordinate(lat, precision)
            upper_lat = round_coordinate(lat + DELTA, precision)

            # Check if this tile overlaps with our area of interest
            if (right_lon > LON_MIN and left_lon < LON_MAX and
                    upper_lat > LAT_MIN and lower_lat < LAT_MAX):
                tiles.append((left_lon, upper_lat, right_lon, lower_lat))

            lat = round_coordinate(lat + DELTA, precision)
        lon = round_coordinate(lon + DELTA, precision)

    return tiles


def point_in_tile(x: float, y: float,
                  tile_bounds: Tuple[float, float, float, float]) -> bool:
    """
    Check if a point (in EPSG:3857 Web Mercator)
    is within a tile's bounds (in WGS84).

    Args:
        x: X coordinate in meters (EPSG:3857)
        y: Y coordinate in meters (EPSG:3857)
        tile_bounds:
            (left_lon, upper_lat, right_lon, lower_lat) in WGS84 degrees

    Returns:
        True if point is in tile
    """
    # Convert from Web Mercator to WGS84
    lon, lat = mercator_to_wgs84(x, y)

    left_lon, upper_lat, right_lon, lower_lat = tile_bounds
    return left_lon <= lon < right_lon and lower_lat <= lat < upper_lat


def append_features_to_file(output_path: Path, features: List[dict],
                            tile_bbox_merc: Tuple[
                                float, float, float, float]
                            = None):
    """
    Append features to a GeoJSON file with proper formatting.
    All float values are rounded to 3 decimal places
    (0.001 precision) to save disk space.

    Args:
        output_path: Path to output file
        features: List of features to append
        tile_bbox_merc: Optional tile bounding box
        (min_x, min_y, max_x, max_y) in Mercator
    """
    if not features:
        return

    # Round all floats in features to 3 decimal places
    features_rounded = round_floats(features, precision=3)

    if not output_path.exists():
        # Create new file with CRS, name, and bbox metadata
        data = {
            "type": "FeatureCollection",
            "name": output_path.stem,
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:EPSG::3857"
                }
            },
            "features": features_rounded
        }

        # Add bbox if provided (in Mercator coordinates)
        if tile_bbox_merc:
            min_x, min_y, max_x, max_y = tile_bbox_merc
            data["bbox"] = [
                round(min_x, 3),
                round(min_y, 3),
                round(max_x, 3),
                round(max_y, 3)
            ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    else:
        # Read and append - we must load existing features
        # This is unavoidable for valid GeoJSON format
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data['features'].extend(features_rounded)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)


def estimate_feature_count(input_file: Path) -> int:
    """
    Estimate the number of features in a GeoJSON file by sampling.

    Reads a small sample of features, measures their average size in bytes,
    and extrapolates to estimate the total feature count.

    Args:
        input_file: Path to input GeoJSON file

    Returns:
        Estimated number of features (or 0 if estimation fails)
    """
    try:
        file_size = input_file.stat().st_size

        MORE = 1000

        with open(input_file, 'rb') as f:
            # Read first 5 features and get position
            features_sample1 = []
            parser1 = ijson.items(f, 'features.item', use_float=True)

            for i, feature in enumerate(parser1):
                if i >= 5:
                    break
                features_sample1.append(feature)

            if len(features_sample1) < 5:
                # File has fewer than 5 features, count exactly
                return len(features_sample1)

            pos_after_5 = f.tell()

            # Read next 10 features and get position
            features_sample2 = []
            for i, feature in enumerate(parser1):
                if i >= MORE:
                    break
                features_sample2.append(feature)

            if len(features_sample2) < MORE:
                # File has fewer than 15 features total
                return len(features_sample1) + len(features_sample2)

            pos_after_more = f.tell()

        # Calculate average bytes per feature from the second sample
        # (more accurate)
        bytes_for_more_features = pos_after_more - pos_after_5
        avg_bytes_per_feature = bytes_for_more_features / MORE

        # Estimate header size (everything before first feature)
        # This includes: {"type":"FeatureCollection","features":[
        # We use position after 5 features and subtract 5 * avg size
        estimated_header_size = pos_after_5 - (5 * avg_bytes_per_feature)

        # Estimate footer size (closing brackets and braces)
        # Typically just: ]} which is about 10-50 bytes, we'll estimate 50
        estimated_footer_size = 50

        # Calculate bytes available for features
        bytes_for_features = (file_size - estimated_header_size
                              - estimated_footer_size)

        # Estimate total features
        estimated_features = int(
            bytes_for_features / avg_bytes_per_feature)

        # At least 15 since we read that many
        return max(estimated_features, 15)

    except Exception as e:
        logger.warning(
            f"  ! Warning: Could not estimate feature count: {e}")
        return 0


def split_input_tile(input_file: Path,
                     output_tiles: List[Tuple[float, float, float, float]],
                     output_dir: Path,
                     tiles_written: Dict[str, int]):
    """
    Split input tile into multiple output tiles using streaming JSON parsing.

    Args:
        input_file: Path to input GeoJSON file
        output_tiles: List of output tile bounds (WGS84)
        output_dir: Directory to save output tiles
        tiles_written: Dictionary tracking feature counts per tile
    """
    logger.info(f"\n  Processing {input_file.name}...")
    
    # Validate file before processing
    try:
        file_size = input_file.stat().st_size
        if file_size == 0:
            logger.error(f"  ✗ File is empty: {input_file.name}")
            return
        
        logger.debug(f"  File size: {file_size / (1024**2):.2f} MB")
        
        # Quick check: verify it starts like JSON
        with open(input_file, 'rb') as f:
            first_bytes = f.read(100)
            if not first_bytes.strip().startswith(b'{'):
                logger.error(f"  ✗ File does not appear to be JSON")
                logger.error(f"     First bytes: {first_bytes[:50]}")
                return
    except Exception as e:
        logger.error(f"  ✗ Cannot validate file: {e}")
        return

    # Estimate feature count for progress reporting
    estimated_count = estimate_feature_count(input_file)
    if estimated_count > 0:
        logger.info(f"  Estimated features: ~{estimated_count:,}")

    # Pre-convert tile bounds to Mercator bounding
    # boxes for fast intersection tests
    # This is done once at startup instead of for every feature
    tile_bboxes_merc = {}
    tile_files = {}

    logger.info(f"  Converting {len(output_tiles)} tiles to Mercator...")

    # Build spatial index: grid cells that point to tiles
    # This avoids checking all 10k tiles for each feature
    grid_size = 100000.0  # 100km grid cells in Mercator
    spatial_index = {}  # (grid_x, grid_y) -> list of tile_bounds

    for idx, bounds in enumerate(output_tiles):
        left_lon, upper_lat, right_lon, lower_lat = bounds

        # Convert WGS84 tile corners to Mercator
        min_x, max_y = wgs84_to_mercator(left_lon, upper_lat)
        max_x, min_y = wgs84_to_mercator(right_lon, lower_lat)

        # Store Mercator bounding box (min_x, min_y, max_x, max_y)
        tile_bbox_merc = (min_x, min_y, max_x, max_y)
        tile_bboxes_merc[bounds] = tile_bbox_merc

        # Add to spatial index - find which grid cells this tile overlaps
        grid_x_min = int(min_x / grid_size)
        grid_x_max = int(max_x / grid_size)
        grid_y_min = int(min_y / grid_size)
        grid_y_max = int(max_y / grid_size)

        for gx in range(grid_x_min, grid_x_max + 1):
            for gy in range(grid_y_min, grid_y_max + 1):
                key = (gx, gy)
                if key not in spatial_index:
                    spatial_index[key] = []
                spatial_index[key].append(bounds)

        # Debug: Print first few tiles
        if idx < 3:
            logger.debug(f"    Tile {idx}: WGS84 ({left_lon:.1f}, "
                         f"{lower_lat:.1f}, {right_lon:.1f}, "
                         f"{upper_lat:.1f}) -> Mercator ({min_x:.0f}, "
                         f"{min_y:.0f}, {max_x:.0f}, {max_y:.0f})")

        # Store file path
        filename = get_output_tile_name(
            left_lon, upper_lat, right_lon, lower_lat)
        tile_files[bounds] = output_dir / filename

    logger.info(
        f"  Built spatial index with {len(spatial_index)} grid cells")

    # Temporary storage for batched writes
    batch_buffers = {bounds: [] for bounds in output_tiles}
    features_processed = 0
    features_with_coords = 0
    features_with_valid_bbox = 0

    logger.info(f"  Streaming features from file...")

    # Ensure BATCH_SIZE is at least 1
    batch_size = max(1, BATCH_SIZE)

    try:
        # Stream parse the JSON file
        with open(input_file, 'rb') as f:
            # Parse features array items one at a time
            features = ijson.items(f, 'features.item', use_float=True)

            # Use tqdm if available
            if HAS_TQDM and estimated_count > 0:
                features = tqdm(features, total=estimated_count,
                                desc="  Processing")

            for feature in features:
                features_processed += 1

                # Get bbox center for this feature
                geometry = feature.get('geometry', {})
                coordinates = geometry.get('coordinates', [])
                if not coordinates:
                    continue

                features_with_coords += 1

                bbox_center = get_bbox_center(coordinates)
                if not bbox_center:
                    if features_processed <= 3:
                        logger.error(f"    Feature {features_processed}:"
                                     f" INVALID BBOX CENTER")
                    continue

                features_with_valid_bbox += 1

                center_x, center_y = bbox_center

                # Debug: Print first few features
                if features_processed <= 3:
                    logger.debug(f"    Feature {features_processed}: "
                                 f"bbox center=({center_x:.0f}, "
                                 f"{center_y:.0f})")
                    logger.debug(f"      Geometry type: "
                                 f"{geometry.get('type')}")

                # Use spatial index to find candidate tiles
                # (single grid cell for point)
                grid_x = int(center_x / grid_size)
                grid_y = int(center_y / grid_size)

                key = (grid_x, grid_y)
                candidate_tiles = spatial_index.get(key, [])

                # Debug: Print candidate count for first few features
                if features_processed <= 3:
                    logger.debug(f"      Found {len(candidate_tiles)}"
                                 f" candidate tiles")

                # Find THE tile this feature belongs to (should be exactly one)
                matched_tile = None
                for tile_bounds in candidate_tiles:
                    tile_bbox_merc = tile_bboxes_merc[tile_bounds]
                    if point_in_tile_bbox(bbox_center, tile_bbox_merc):
                        matched_tile = tile_bounds
                        break  # Found it, stop searching

                if matched_tile:
                    batch_buffers[matched_tile].append(feature)

                    # Debug: Print match for first few features
                    if features_processed <= 3:
                        logger.debug(f"      Matched tile: {matched_tile}")
                elif features_processed < 1:
                    logger.warning(
                        f"      WARNING: No features instide output tile!")

                # Flush batches when they get large enough
                for tile_bounds in list(batch_buffers.keys()):
                    batch = batch_buffers[tile_bounds]
                    if len(batch) >= batch_size:
                        output_path = tile_files[tile_bounds]
                        tile_bbox_merc = tile_bboxes_merc[tile_bounds]
                        append_features_to_file(
                            output_path, batch, tile_bbox_merc)
                        # Update counter
                        filename = output_path.name
                        tiles_written[filename] = tiles_written.get(
                            filename, 0) + len(batch)
                        # Clear batch
                        batch_buffers[tile_bounds] = []

        if not HAS_TQDM:
            logger.info(f"  Processed: {features_processed:,}"
                        f" features (complete)")

        logger.info(f"  Statistics:")
        logger.info(
            f"    Total features processed: {features_processed:,}")
        logger.info(
            f"    Features with coordinates: {features_with_coords:,}")
        logger.info(f"    Features with valid bbox:"
                    f" {features_with_valid_bbox:,}")

        if estimated_count > 0:
            accuracy = (features_processed / estimated_count) * 100
            logger.info(
                f"  Estimation accuracy: {accuracy:.1f}%"
                f" (estimated {estimated_count:,},"
                f" actual {features_processed:,})")

        # Flush remaining batches
        logger.info(f"  Writing remaining features...")
        total_features_in_batches = sum(len(batch)
                                        for batch in
                                        batch_buffers.values())
        logger.info(f"  Total features in batches before "
                    f"flush: {total_features_in_batches}")

        for tile_bounds, batch in batch_buffers.items():
            if batch:
                output_path = tile_files[tile_bounds]
                tile_bbox_merc = tile_bboxes_merc[tile_bounds]
                logger.debug(f"    Flushing {len(batch)} "
                             f"features to {output_path.name}")
                append_features_to_file(output_path, batch, tile_bbox_merc)
                # Update counter
                filename = output_path.name
                tiles_written[filename] = tiles_written.get(
                    filename, 0) + len(batch)

        # Free memory
        del batch_buffers

        logger.debug(f"  ✓ Completed processing {input_file.name}")

    except json.JSONDecodeError as e:
        logger.error(f"  ✗ JSON parsing error in {input_file.name}:")
        logger.error(f"     {e}")
        logger.error(f"     Successfully processed {features_processed:,} features before error")
        logger.error(f"     Position in file: character {e.pos}")
        
        # Show context around error if in debug mode
        if logger.getEffectiveLevel() == logging.DEBUG:
            try:
                with open(input_file, 'rb') as f:
                    f.seek(max(0, e.pos - 200))
                    context = f.read(400).decode('utf-8', errors='replace')
                    logger.debug(f"     Context around error:")
                    logger.debug(f"     ...{context}...")
            except Exception:
                pass
        
        # Try to give context about which feature failed
        if features_processed > 0:
            logger.error(f"     Error likely in feature #{features_processed + 1}")
        
        logger.error(f"     File may be corrupted or have encoding issues")
        logger.info(f"     Suggestion: Try re-downloading this file")
        
        # Show what we managed to process
        if features_with_valid_bbox > 0:
            logger.info(f"     Partial results: Processed {features_with_valid_bbox:,} valid features")
            logger.info(f"     Continuing with next file...")
        
    except ijson.JSONError as e:
        logger.error(f"  ✗ ijson parsing error in {input_file.name}:")
        logger.error(f"     {e}")
        logger.error(f"     Successfully processed {features_processed:,} features before error")
        logger.error(f"     This may indicate a malformed feature in the GeoJSON")
        
        # Show which feature failed
        if features_processed > 0:
            logger.error(f"     Failed at feature #{features_processed + 1}")
        
        logger.info(f"     Try using: ijson backend 'python' for better error messages")
        logger.info(f"     Continuing with next file...")
    
    except Exception as e:
        logger.error(f"  ✗ Error processing {input_file.name}: {e}")
        logger.error(f"     Processed {features_processed:,} features before error")
        import traceback
        if logger.getEffectiveLevel() == logging.DEBUG:
            traceback.print_exc()


def download_country_boundary(urls: List[str], cache_file: Path
                              ) -> Optional[Dict]:
    """Download and cache country boundary data."""
    if cache_file.exists():
        logger.debug(f"Using cached country data: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    logger.info("Downloading country boundary data...")

    for url in urls:
        logger.debug(f"Trying URL: {url}")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            logger.debug("Download successful")
            break
        except Exception as e:
            logger.debug(f"Failed: {e}")
            continue
    else:
        logger.error("All download sources failed")
        return None

    logger.debug("Extracting shapefile...")
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            shp_files = [f for f in z.namelist() if f.endswith('.shp')]
            if not shp_files:
                logger.error("No .shp file found in ZIP")
                return None

            shp_name = shp_files[0]
            base_name = shp_name.replace('.shp', '')

            temp_dir = Path(CACHE_DIR) / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                filename = base_name + ext
                if filename in z.namelist():
                    z.extract(filename, temp_dir)

            shp_path = temp_dir / shp_name

            logger.debug("Converting to GeoJSON using OGR...")
            driver = ogr.GetDriverByName('ESRI Shapefile')
            datasource = driver.Open(str(shp_path), 0)

            if datasource is None:
                logger.error(f"Could not open shapefile: {shp_path}")
                return None

            layer = datasource.GetLayer()
            features = []

            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom:
                    geom_json = json.loads(geom.ExportToJson())
                else:
                    geom_json = None

                attributes = {}
                for i in range(feature.GetFieldCount()):
                    field_name = feature.GetFieldDefnRef(i).GetName()
                    field_value = feature.GetField(i)
                    attributes[field_name] = field_value

                features.append({
                    "type": "Feature",
                    "geometry": geom_json,
                    "properties": attributes
                })

            geojson = {
                "type": "FeatureCollection",
                "features": features
            }

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(geojson, f)

            logger.debug(f"Cached to: {cache_file}")
            return geojson

    except Exception as e:
        logger.error(f"Error extracting: {e}")
        return None


def load_country_bbox(name: str | None = None,
                      iso2: str | None = None,
                      iso3: str | None = None):
    """
    Load country bounding box and geometry from Natural Earth data.
    Parameter:
        - name: Country Name
        - iso2: Country ISO two-letter code
        - iso3: Country ISO three-letter code

    Returns:
        Tuple of (bbox, geometry) where:
        - bbox: (min_lon, min_lat, max_lon, max_lat)
        - geometry: OGR Geometry object (Polygon or MultiPolygon)
        Returns None if not found
    """
    if not HAS_REQUESTS or not HAS_OGR:
        logger.error(
            "Country boundaries require 'requests' and 'gdal' (osgeo) modules")
        logger.error("Install with: pip install requests gdal")
        return None

    cache_file = Path(CACHE_DIR) / "countries_10m.geojson"

    geojson = download_country_boundary(DETAILED_COUNTRY_URLS,
                                        cache_file)
    if not geojson:
        return None

    for feature in geojson.get('features', []):
        properties = feature.get('properties', {})
        names = [
            properties.get('NAME', ''),
            properties.get('NAME_LONG', ''),
            properties.get('ADMIN', ''),
            properties.get('NAME_EN', '')
        ]
        isotwo = [
            properties.get('ISO_A2_EH', ''),
            properties.get('ISO_A2', ''),  # Fallback for countries with -99 in _EH
        ]
        isothree = [
            properties.get('ISO_A3_EH', ''),
            properties.get('ISO_A3', ''),  # Fallback for countries with -99 in _EH
        ]

        if (
            (name and any(name.lower() in db_name.lower() for db_name in names))
            or
            (iso2 and any(iso2.upper() == db_iso2.upper() and db_iso2 != '-99' 
                         for db_iso2 in isotwo))
            or
            (iso3 and any(iso3.upper() == db_iso3.upper() and db_iso3 != '-99' 
                         for db_iso3 in isothree))
        ):
            country_name = properties.get('NAME', 'Unknown')
            logger.info(f"Found country: {country_name}")

            geom_json = json.dumps(feature['geometry'])
            geom = ogr.CreateGeometryFromJson(geom_json)

            envelope = geom.GetEnvelope()
            # OGR envelope is (minX, maxX, minY, maxY)
            bbox = (envelope[0], envelope[2], envelope[1], envelope[3])
            logger.info(f"Country bbox: {bbox[0]:.3f},"
                        f" {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}")
            
            return bbox, geom, country_name

    if name:
        logger.error(f"Country name '{name}' not found")
    elif iso2:
        logger.info(f"Country code '{iso2}' not found")
    elif iso3:
        logger.info(f"Country code '{iso3}' not found")
    return None


def filter_tiles_by_country(tiles: List[Tuple[float, float, float, float]],
                            country_geom) -> List[Tuple[float, float,
                                                        float, float]]:
    """
    Filter tiles to only those that intersect with country geometry.
    
    Args:
        tiles: List of tile bounds (left_lon, upper_lat, right_lon, lower_lat)
        country_geom: OGR Geometry object for country
    
    Returns:
        Filtered list of tiles that intersect with country
    """
    if country_geom is None:
        return tiles
    
    logger.info(f"Filtering {len(tiles)} tiles by country intersection...")
    
    intersecting_tiles = []
    
    for left_lon, upper_lat, right_lon, lower_lat in tiles:
        # Create tile polygon in WGS84
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(left_lon, lower_lat)
        ring.AddPoint(right_lon, lower_lat)
        ring.AddPoint(right_lon, upper_lat)
        ring.AddPoint(left_lon, upper_lat)
        ring.AddPoint(left_lon, lower_lat)  # Close ring
        
        tile_geom = ogr.Geometry(ogr.wkbPolygon)
        tile_geom.AddGeometry(ring)
        
        # Test intersection
        if country_geom.Intersects(tile_geom):
            intersecting_tiles.append((left_lon, upper_lat,
                                      right_lon, lower_lat))
    
    logger.info(f"  {len(intersecting_tiles)} tiles intersect with country")
    logger.info(f"  {len(tiles) - len(intersecting_tiles)} tiles filtered out")
    
    return intersecting_tiles


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GlobalBuildingAtlas Downloader and Tiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use bounding box
  %(prog)s --bbox 5.0 45.0 15.0 55.0

  # Use country name
  %(prog)s --country Germany

  # Use ISO 2-letter code
  %(prog)s --iso2 DE

  # Use ISO 3-letter code
  %(prog)s --iso3 DEU

  # With verbose output
  %(prog)s --country France --verbose

  # With debug output
  %(prog)s --bbox 10.0 50.0 11.0 51.0 --debug

  # Show version
  %(prog)s --version
        '''
    )
    
    # Add version argument
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )

    # Area specification (mutually exclusive)
    area_group = parser.add_mutually_exclusive_group(required=True)
    area_group.add_argument(
        '--bbox',
        nargs=4,
        metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'),
        type=float,
        help='Bounding box: min_lon min_lat max_lon max_lat (in degrees)'
    )
    area_group.add_argument(
        '--country',
        type=str,
        metavar='NAME',
        help='Country name (e.g., "Germany", "France")'
    )
    area_group.add_argument(
        '--iso2',
        type=str,
        metavar='CODE',
        help='Country ISO 2-letter code (e.g., "DE", "FR")'
    )
    area_group.add_argument(
        '--iso3',
        type=str,
        metavar='CODE',
        help='Country ISO 3-letter code (e.g., "DEU", "FRA")'
    )

    # Logging (mutually exclusive)
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (INFO level)'
    )
    log_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output (DEBUG level)'
    )

    # Optional parameters
    parser.add_argument(
        '--delta',
        type=float,
        default=0.10,
        metavar='DEGREES',
        help='Tile size in degrees (default: 0.10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        metavar='N',
        help='Batch size for writing features (default: 1000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='GBA_tiles',
        metavar='DIR',
        help='Output directory (default: GBA_tiles)'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='GBA_temp',
        metavar='DIR',
        help='Temporary directory (default: GBA_temp)'
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False, debug: bool = False):
    """Setup logging configuration."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def main():
    """Main execution function."""
    args = parse_args()
    setup_logging(args.verbose, args.debug)

    # Set global configuration from arguments
    global DELTA, BATCH_SIZE, OUTPUT_DIR, TEMP_DIR
    global LON_MIN, LAT_MIN, LON_MAX, LAT_MAX

    DELTA = args.delta
    BATCH_SIZE = args.batch_size
    OUTPUT_DIR = args.output_dir
    TEMP_DIR = args.temp_dir

    # Determine area of interest
    country_geom = None
    country_name = None
    if args.bbox:
        LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = args.bbox
        logger.info(f"Using bounding box: {LON_MIN}°E to"
                    f" {LON_MAX}°E, {LAT_MIN}°N to {LAT_MAX}°N")
    elif args.country or args.iso2 or args.iso3:
        search_term = args.country or args.iso2 or args.iso3
        if args.country:
            result = load_country_bbox(name=args.country)
        elif args.iso2:
            result = load_country_bbox(iso2=args.iso2)
        elif args.iso3:
            result = load_country_bbox(iso3=args.iso3)
        if result is None:
            logger.critical(
                f"Could not load boundaries for: {search_term}")
            sys.exit(1)
        bbox, country_geom, country_name = result
        LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = bbox
        
        # Prominent country message
        logger.info("=" * 60)
        logger.info(f"  Processing country: {country_name}")
        logger.info("=" * 60)
        logger.info(f"Bounding box: {LON_MIN:.3f}°E to {LON_MAX:.3f}°E,"
                    f" {LAT_MIN:.3f}°N to {LAT_MAX:.3f}°N")

    logger.info(f"Tile size: {DELTA}°")
    logger.info(f"Batch size: {max(1, BATCH_SIZE)} features")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Create directories
    temp_dir = Path(TEMP_DIR)
    output_dir = Path(OUTPUT_DIR)
    temp_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Calculate which input tiles we need
    logger.info("Step 1: Determining required input tiles...")
    input_tiles = get_input_tile_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    logger.info(f"Need {len(input_tiles)} input tile(s)")

    # Calculate output tile structure
    logger.info("Step 2: Calculating output tile grid...")
    output_tiles = get_output_tiles()
    total_output_tiles = len(output_tiles)
    logger.info(f"Generated {total_output_tiles} tiles in bounding box")
    
    # Filter tiles by country geometry if using --country
    if country_geom is not None:
        output_tiles = filter_tiles_by_country(output_tiles, country_geom)
        logger.info(f"Will create {len(output_tiles)} output tile(s) "
                   f"(filtered by country intersection)")
    else:
        logger.info(f"Will create up to {total_output_tiles} output tile(s)")

    # Download input tiles
    logger.info("Step 3: Downloading input tiles...")
    downloaded_files = []
    for idx, (left_lon, upper_lat, right_lon, lower_lat
              ) in enumerate(input_tiles, 1):
        filename = get_input_filename(
            left_lon, upper_lat, right_lon, lower_lat)
        region_guess = guess_region(
            left_lon, upper_lat, right_lon, lower_lat)
        if download_input_tile(filename, temp_dir, idx, len(input_tiles),
                               region_guess=region_guess):
            downloaded_files.append(temp_dir / filename)

    if not downloaded_files:
        logger.critical("No files were downloaded successfully.")
        sys.exit(1)

    # Process and split tiles
    logger.info("Step 4: Splitting tiles...")
    logger.info(f"Processing {len(downloaded_files)} input file(s)"
                f" using streaming...")

    tiles_written = {}  # Track feature counts per output tile

    for input_file in downloaded_files:
        split_input_tile(input_file, output_tiles, output_dir,
                         tiles_written)

    # Summary
    logger.info("=" * 50)
    logger.info("Complete!")
    output_files = list(output_dir.glob("*.geojson"))
    logger.info(
        f"Created {len(output_files)} output tile(s) in {OUTPUT_DIR}/")

    # Show feature counts
    if tiles_written:
        files_with_features = [(k, v)
                               for k, v in tiles_written.items() if v > 0]
        logger.info(f"Files with features: {len(files_with_features)}")
        logger.debug("Feature counts per tile:")
        for filename in sorted(tiles_written.keys())[:10]:
            count = tiles_written[filename]
            if count > 0:
                logger.debug(f"  {filename}: {count} features")

    logger.debug(f"Input files stored in {TEMP_DIR}/")


if __name__ == "__main__":
    main()
