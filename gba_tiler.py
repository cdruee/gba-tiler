#!/usr/bin/env python3
"""
GlobalBuildingAtlas Downloader and Tiler

Downloads GeoJSON files from GlobalBuildingAtlas via rsync and splits them
into smaller tiles with configurable resolution.

Streaming version: Uses ijson for memory-efficient JSON parsing.

Note: Input files use EPSG:3857 (Web Mercator) coordinate system with coordinates
in meters. Output tiles are defined in WGS84 (EPSG:4326) lat/lon degrees.
The script automatically converts between coordinate systems.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Dict
import math

try:
    import ijson
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
    print("Warning: tqdm not found, progress bars will be disabled")

# Earth radius for Web Mercator projection
EARTH_RADIUS = 6378137.0  # meters


class RoundingFloat(float):
    """Float subclass that rounds to 3 decimal places when serializing to JSON."""
    __repr__ = lambda self: f"{self:.3f}".rstrip('0').rstrip('.')


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
LON_MIN = 5.0   # Minimum longitude of interest
LON_MAX = 15.0  # Maximum longitude of interest

# rsync configuration
RSYNC_SERVER = "rsync://m1782307:m1782307@dataserv.ub.tum.de/m1782307/LoD1/europe/"
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
    lat = (2.0 * math.atan(math.exp(y / EARTH_RADIUS)) - math.pi / 2.0) * (180.0 / math.pi)
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
    y = EARTH_RADIUS * math.log(math.tan(math.pi / 4.0 + lat * (math.pi / 180.0) / 2.0))
    return x, y


def get_feature_bbox_fast(coordinates) -> Tuple[float, float, float, float]:
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
        if not isinstance(first, (list, tuple)) and not isinstance(second, (list, tuple)):
            try:
                x = float(first)
                y = float(second)
                
                # Validate these are reasonable coordinate values (in meters for Web Mercator)
                if -20037509 <= x <= 20037509 and -20037509 <= y <= 20037509:
                    # This is a valid coordinate pair
                    if min_x > x: min_x = x
                    if max_x < x: max_x = x
                    if min_y > y: min_y = y
                    if max_y < y: max_y = y
                    found = True
                    continue
            except (ValueError, TypeError):
                pass
        
        # If not a coordinate pair, add nested items to stack
        for item in coords:
            if isinstance(item, (list, tuple)):
                stack.append(item)
    
    if found:
        return (min_x, min_y, max_x, max_y)
    return None


def get_bbox_center(coordinates) -> Tuple[float, float]:
    """
    Get the center of the bounding box of GeoJSON coordinates.
    This is the arithmetic mean: (mean(min_x, max_x), mean(min_y, max_y))
    
    Args:
        coordinates: GeoJSON coordinates array
    
    Returns:
        Tuple of (x, y) representing bbox center, or None if no valid coordinates
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
        if not isinstance(first, (list, tuple)) and not isinstance(second, (list, tuple)):
            try:
                x = float(first)
                y = float(second)
                
                # Validate Web Mercator range
                if -20037509 <= x <= 20037509 and -20037509 <= y <= 20037509:
                    if min_x > x: min_x = x
                    if max_x < x: max_x = x
                    if min_y > y: min_y = y
                    if max_y < y: max_y = y
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
        return (center_x, center_y)
    
    return None


def point_in_tile_bbox(point: Tuple[float, float], 
                       tile_bbox_merc: Tuple[float, float, float, float]) -> bool:
    """
    Check if a point is inside a tile bounding box.
    
    Args:
        point: (x, y) in Mercator coordinates
        tile_bbox_merc: Tile bounding box (min_x, min_y, max_x, max_y) in Mercator
    
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


def get_output_tile_name(left_lon: float, upper_lat: float, right_lon: float, lower_lat: float) -> str:
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

    return f"{lon_left_str}_{lat_upper_str}_{lon_right_str}_{lat_lower_str}_lod1.geojson"


def get_input_tile_bounds(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> List[Tuple[float, float, float, float]]:
    """
    Calculate which 5x5 degree input tiles are needed to cover the area of interest.

    Returns:
        List of (left_lon, upper_lat, right_lon, lower_lat) tuples
    """
    tiles = []

    # Input tiles are 5x5 degrees
    input_tile_size = 5

    # Find all 5x5 degree tiles that overlap with our area
    for lon in range(int(math.floor(lon_min / input_tile_size) * input_tile_size),
                     int(math.ceil(lon_max / input_tile_size) * input_tile_size),
                     input_tile_size):
        for lat in range(int(math.floor(lat_min / input_tile_size) * input_tile_size),
                        int(math.ceil(lat_max / input_tile_size) * input_tile_size),
                        input_tile_size):
            left_lon = lon
            right_lon = lon + input_tile_size
            lower_lat = lat
            upper_lat = lat + input_tile_size

            tiles.append((left_lon, upper_lat, right_lon, lower_lat))

    return tiles


def get_input_filename(left_lon: float, upper_lat: float, right_lon: float, lower_lat: float) -> str:
    """
    Generate input tile filename from bounds.
    """
    lon_left_str = get_input_coordinate_string(left_lon, True)
    lat_upper_str = get_input_coordinate_string(upper_lat, False)
    lon_right_str = get_input_coordinate_string(right_lon, True)
    lat_lower_str = get_input_coordinate_string(lower_lat, False)

    return f"{lon_left_str}_{lat_upper_str}_{lon_right_str}_{lat_lower_str}.geojson"


def download_input_tile(filename: str, temp_dir: Path, file_num: int, total_files: int) -> bool:
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
        print(f"  File already exists: {filename} (file {file_num} of {total_files})")
        return True

    print(f"  Downloading {filename} (file {file_num} of {total_files})...")

    # Construct rsync URL without password in URL (use environment variable instead)
    rsync_url = f"rsync://m1782307@dataserv.ub.tum.de/m1782307/LoD1/europe/{filename}"

    # Set password via environment variable
    env = os.environ.copy()
    env["RSYNC_PASSWORD"] = "m1782307"

    try:
        # Use --progress to show progress bar, --no-motd to suppress message of the day
        result = subprocess.run(
            ["rsync", "-av", "--progress", "--no-motd", rsync_url, str(output_path)],
            capture_output=False,  # Don't capture so progress bar shows
            text=True,
            timeout=300,
            env=env
        )

        if result.returncode == 0:
            print(f"  ✓ Downloaded {filename}")
            return True
        else:
            print(f"  ✗ Failed to download {filename}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout downloading {filename}")
        return False
    except FileNotFoundError:
        print(f"  ✗ rsync command not found. Please install rsync.")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Error downloading {filename}: {e}")
        return False


def get_output_tiles() -> List[Tuple[float, float, float, float]]:
    """
    Calculate all output tile boundaries based on DELTA and the area of interest.

    Returns:
        List of (left_lon, upper_lat, right_lon, lower_lat) tuples
    """
    tiles = []
    
    # Get rounding precision based on DELTA
    precision = get_delta_precision()

    # Align to grid
    lon_start = round_coordinate(math.floor(LON_MIN / DELTA) * DELTA, precision)
    lon_end = round_coordinate(math.ceil(LON_MAX / DELTA) * DELTA, precision)
    lat_start = round_coordinate(math.floor(LAT_MIN / DELTA) * DELTA, precision)
    lat_end = round_coordinate(math.ceil(LAT_MAX / DELTA) * DELTA, precision)

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


def point_in_tile(x: float, y: float, tile_bounds: Tuple[float, float, float, float]) -> bool:
    """
    Check if a point (in EPSG:3857 Web Mercator) is within a tile's bounds (in WGS84).

    Args:
        x: X coordinate in meters (EPSG:3857)
        y: Y coordinate in meters (EPSG:3857)
        tile_bounds: (left_lon, upper_lat, right_lon, lower_lat) in WGS84 degrees

    Returns:
        True if point is in tile
    """
    # Convert from Web Mercator to WGS84
    lon, lat = mercator_to_wgs84(x, y)

    left_lon, upper_lat, right_lon, lower_lat = tile_bounds
    return (left_lon <= lon < right_lon and lower_lat <= lat < upper_lat)


def feature_intersects_tile(feature: dict, tile_bbox_merc: Tuple[float, float, float, float]) -> bool:
    """
    Check if a GeoJSON feature (in EPSG:3857) intersects with a tile (in Mercator).
    Uses bounding box intersection for speed.

    Args:
        feature: GeoJSON feature with coordinates in EPSG:3857 (Web Mercator, meters)
        tile_bbox_merc: Tile bounding box (min_x, min_y, max_x, max_y) in Mercator

    Returns:
        True if feature intersects tile
    """
    geometry = feature.get('geometry', {})
    coordinates = geometry.get('coordinates', [])

    if not coordinates:
        return False

    # Calculate feature bounding box
    feature_bbox = get_feature_bbox_fast(coordinates)
    if not feature_bbox:
        return False
    
    # Fast bounding box intersection test
    return bbox_intersects_tile(feature_bbox, tile_bbox_merc)


def append_features_to_file(output_path: Path, features: List[dict], tile_bbox_merc: Tuple[float, float, float, float] = None):
    """
    Append features to a GeoJSON file with proper formatting.
    All float values are rounded to 3 decimal places (0.001 precision) to save disk space.

    Args:
        output_path: Path to output file
        features: List of features to append
        tile_bbox_merc: Optional tile bounding box (min_x, min_y, max_x, max_y) in Mercator
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

        # Calculate average bytes per feature from the second sample (more accurate)
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
        bytes_for_features = file_size - estimated_header_size - estimated_footer_size

        # Estimate total features
        estimated_features = int(bytes_for_features / avg_bytes_per_feature)

        return max(estimated_features, 15)  # At least 15 since we read that many

    except Exception as e:
        print(f"  ! Warning: Could not estimate feature count: {e}")
        return 0


def split_input_tile(input_file: Path, output_tiles: List[Tuple[float, float, float, float]],
                     output_dir: Path, tiles_written: Dict[str, int]):
    """
    Split an input tile into multiple output tiles using streaming JSON parsing.

    Args:
        input_file: Path to input GeoJSON file
        output_tiles: List of output tile bounds (WGS84)
        output_dir: Directory to save output tiles
        tiles_written: Dictionary tracking feature counts per tile
    """
    print(f"\n  Processing {input_file.name}...")

    # Estimate feature count for progress reporting
    estimated_count = estimate_feature_count(input_file)
    if estimated_count > 0:
        print(f"  Estimated features: ~{estimated_count:,}")

    # Pre-convert tile bounds to Mercator bounding boxes for fast intersection tests
    # This is done once at startup instead of for every feature
    tile_bboxes_merc = {}
    tile_files = {}
    
    print(f"  Converting {len(output_tiles)} tiles to Mercator...")
    
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
            print(f"    Tile {idx}: WGS84 ({left_lon:.1f}, {lower_lat:.1f}, {right_lon:.1f}, {upper_lat:.1f}) -> Mercator ({min_x:.0f}, {min_y:.0f}, {max_x:.0f}, {max_y:.0f})")
        
        # Store file path
        filename = get_output_tile_name(left_lon, upper_lat, right_lon, lower_lat)
        tile_files[bounds] = output_dir / filename
    
    print(f"  Built spatial index with {len(spatial_index)} grid cells")

    # Temporary storage for batched writes
    batch_buffers = {bounds: [] for bounds in output_tiles}
    features_processed = 0
    features_with_coords = 0
    features_with_valid_bbox = 0

    print(f"  Streaming features from file...")

    # Ensure BATCH_SIZE is at least 1
    batch_size = max(1, BATCH_SIZE)

    try:
        # Stream parse the JSON file
        with open(input_file, 'rb') as f:
            # Parse features array items one at a time
            features = ijson.items(f, 'features.item', use_float=True)

            # Use tqdm if available
            if HAS_TQDM and estimated_count > 0:
                features = tqdm(features, total=estimated_count, desc="  Processing")

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
                        print(f"    Feature {features_processed}: INVALID BBOX CENTER")
                    continue

                features_with_valid_bbox += 1
                
                center_x, center_y = bbox_center
                
                # Debug: Print first few features
                if features_processed <= 3:
                    print(f"    Feature {features_processed}: bbox center=({center_x:.0f}, {center_y:.0f})")
                    print(f"      Geometry type: {geometry.get('type')}")

                # Use spatial index to find candidate tiles (single grid cell for point)
                grid_x = int(center_x / grid_size)
                grid_y = int(center_y / grid_size)
                
                key = (grid_x, grid_y)
                candidate_tiles = spatial_index.get(key, [])
                
                # Debug: Print candidate count for first few features
                if features_processed <= 3:
                    print(f"      Found {len(candidate_tiles)} candidate tiles")

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
                        print(f"      Matched tile: {matched_tile}")
                elif features_processed <= 3:
                    print(f"      WARNING: No matching tile found!")

                # Flush batches when they get large enough
                for tile_bounds in list(batch_buffers.keys()):
                    batch = batch_buffers[tile_bounds]
                    if len(batch) >= batch_size:
                        output_path = tile_files[tile_bounds]
                        tile_bbox_merc = tile_bboxes_merc[tile_bounds]
                        append_features_to_file(output_path, batch, tile_bbox_merc)
                        # Update counter
                        filename = output_path.name
                        tiles_written[filename] = tiles_written.get(filename, 0) + len(batch)
                        # Clear batch
                        batch_buffers[tile_bounds] = []

        if not HAS_TQDM:
            print(f"  Processed: {features_processed:,} features (complete)")

        print(f"  Statistics:")
        print(f"    Total features processed: {features_processed:,}")
        print(f"    Features with coordinates: {features_with_coords:,}")
        print(f"    Features with valid bbox: {features_with_valid_bbox:,}")

        if estimated_count > 0:
            accuracy = (features_processed / estimated_count) * 100
            print(f"  Estimation accuracy: {accuracy:.1f}% (estimated {estimated_count:,}, actual {features_processed:,})")

        # Flush remaining batches
        print(f"  Writing remaining features...")
        total_features_in_batches = sum(len(batch) for batch in batch_buffers.values())
        print(f"  Total features in batches before flush: {total_features_in_batches}")
        
        for tile_bounds, batch in batch_buffers.items():
            if batch:
                output_path = tile_files[tile_bounds]
                tile_bbox_merc = tile_bboxes_merc[tile_bounds]
                print(f"    Flushing {len(batch)} features to {output_path.name}")
                append_features_to_file(output_path, batch, tile_bbox_merc)
                # Update counter
                filename = output_path.name
                tiles_written[filename] = tiles_written.get(filename, 0) + len(batch)

        # Free memory
        del batch_buffers

        print(f"  ✓ Completed processing {input_file.name}")

    except Exception as e:
        print(f"  ✗ Error processing {input_file.name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main execution function."""
    print("GlobalBuildingAtlas Downloader and Tiler (Streaming)")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Area: {LON_MIN}°E to {LON_MAX}°E, {LAT_MIN}°N to {LAT_MAX}°N")
    print(f"  Tile size: {DELTA}°")
    print(f"  Batch size: {max(1, BATCH_SIZE)} features")
    print()

    # Create directories
    temp_dir = Path(TEMP_DIR)
    output_dir = Path(OUTPUT_DIR)
    temp_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Calculate which input tiles we need
    print("Step 1: Determining required input tiles...")
    input_tiles = get_input_tile_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    print(f"  Need {len(input_tiles)} input tile(s)")

    # Calculate output tile structure
    print("\nStep 2: Calculating output tile grid...")
    output_tiles = get_output_tiles()
    total_output_tiles = len(output_tiles)
    print(f"  Will create up to {total_output_tiles} output tile(s)")

    # Download input tiles
    print("\nStep 3: Downloading input tiles...")
    downloaded_files = []
    for idx, (left_lon, upper_lat, right_lon, lower_lat) in enumerate(input_tiles, 1):
        filename = get_input_filename(left_lon, upper_lat, right_lon, lower_lat)
        if download_input_tile(filename, temp_dir, idx, len(input_tiles)):
            downloaded_files.append(temp_dir / filename)

    if not downloaded_files:
        print("\n✗ No files were downloaded successfully.")
        sys.exit(1)

    # Process and split tiles
    print(f"\nStep 4: Splitting tiles...")
    print(f"Processing {len(downloaded_files)} input file(s) using streaming...")

    tiles_written = {}  # Track feature counts per output tile

    for input_file in downloaded_files:
        split_input_tile(input_file, output_tiles, output_dir, tiles_written)

    # Summary
    print("\n" + "=" * 50)
    print("Complete!")
    output_files = list(output_dir.glob("*.geojson"))
    print(f"Created {len(output_files)} output tile(s) in {OUTPUT_DIR}/")

    # Show feature counts
    if tiles_written:
        print(f"\nFeature counts per tile:")
        for filename in sorted(tiles_written.keys()):
            count = tiles_written[filename]
            print(f"  {filename}: {count} features")

    print(f"\nInput files stored in {TEMP_DIR}/")


if __name__ == "__main__":
    main()
    