#!/usr/bin/env python3
"""
GlobalBuildingAtlas Downloader and Tiler

Downloads GeoJSON files from GlobalBuildingAtlas via rsync and splits them
into smaller tiles with configurable resolution.

Streaming version: Uses ijson for memory-efficient JSON parsing.
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
import tqdm

# Configuration constants
DELTA = 0.25  # Tile size in degrees
LAT_MIN = 45.0  # Minimum latitude of interest
LAT_MAX = 55.0  # Maximum latitude of interest
LON_MIN = 5.0   # Minimum longitude of interest
LON_MAX = 15.0  # Maximum longitude of interest

# rsync configuration
RSYNC_SERVER = "rsync://m1782307:m1782307@dataserv.ub.tum.de/m1782307/LoD1/europe/"
OUTPUT_DIR = "GBA_tiles"
TEMP_DIR = "GBA_temp"

# Memory optimization: batch size for writing features
BATCH_SIZE = 0


def get_coordinate_string(value: float, is_longitude: bool) -> str:
    """
    Format coordinate as string following the naming convention.

    Args:
        value: Coordinate value
        is_longitude: True for longitude, False for latitude

    Returns:
        Formatted string (e.g., 'e005' or 'n50')
    """
    if is_longitude:
        direction = 'e' if value >= 0 else 'w'
        digits = 5  # For output tiles
        abs_val = abs(int(value))
        return f"{direction}{abs_val:0{digits}d}"
    else:
        direction = 'n' if value >= 0 else 's'
        digits = 4  # For output tiles
        abs_val = abs(int(value))
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

    return f"{lon_left_str}_{lat_upper_str}_{lon_right_str}_{lat_lower_str}.geojson"


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

    # Align to grid
    lon_start = math.floor(LON_MIN / DELTA) * DELTA
    lon_end = math.ceil(LON_MAX / DELTA) * DELTA
    lat_start = math.floor(LAT_MIN / DELTA) * DELTA
    lat_end = math.ceil(LAT_MAX / DELTA) * DELTA

    lon = lon_start
    while lon < lon_end:
        lat = lat_start
        while lat < lat_end:
            left_lon = lon
            right_lon = lon + DELTA
            lower_lat = lat
            upper_lat = lat + DELTA

            # Check if this tile overlaps with our area of interest
            if (right_lon > LON_MIN and left_lon < LON_MAX and
                upper_lat > LAT_MIN and lower_lat < LAT_MAX):
                tiles.append((left_lon, upper_lat, right_lon, lower_lat))

            lat += DELTA
        lon += DELTA

    return tiles


def point_in_tile(lon: float, lat: float, tile_bounds: Tuple[float, float, float, float]) -> bool:
    """
    Check if a point is within a tile's bounds.

    Args:
        lon: Longitude of point
        lat: Latitude of point
        tile_bounds: (left_lon, upper_lat, right_lon, lower_lat)

    Returns:
        True if point is in tile
    """
    left_lon, upper_lat, right_lon, lower_lat = tile_bounds
    return (left_lon <= lon < right_lon and lower_lat <= lat < upper_lat)


def feature_intersects_tile(feature: dict, tile_bounds: Tuple[float, float, float, float]) -> bool:
    """
    Check if a GeoJSON feature intersects with a tile.

    Args:
        feature: GeoJSON feature
        tile_bounds: (left_lon, upper_lat, right_lon, lower_lat)

    Returns:
        True if feature intersects tile
    """
    geometry = feature.get('geometry', {})
    coordinates = geometry.get('coordinates', [])

    if not coordinates:
        return False

    # For simplicity, check if any coordinate is within the tile
    # This is a conservative approach - includes features that touch the tile
    def check_coords(coords):
        if not coords:
            return False

        # Handle Decimal objects from ijson
        if isinstance(coords, Decimal):
            return False

        # Check if this is a coordinate pair [lon, lat]
        # Coordinates can be int, float, or Decimal (from ijson)
        if len(coords) >= 2:
            first = coords[0]
            # Check if first element is a number (not a list)
            if isinstance(first, (int, float, Decimal)):
                # This is a coordinate pair
                lon = float(coords[0])
                lat = float(coords[1])
                return point_in_tile(lon, lat, tile_bounds)

        # Otherwise, recurse into nested lists
        try:
            return any(check_coords(c) for c in coords)
        except (TypeError, AttributeError):
            return False

    return check_coords(coordinates)


def flush_batch_to_file(output_path: Path, features: List[dict], tiles_written: Dict[str, int]):
    """
    Write a batch of features to a file.

    Args:
        output_path: Path to output file
        features: List of features to write
        tiles_written: Dictionary tracking feature counts
    """
    if not features:
        return

    filename = output_path.name

    try:
        if not output_path.exists():
            # Create new file
            data = {
                "type": "FeatureCollection",
                "features": features
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            if filename not in tiles_written:
                tiles_written[filename] = 0
        else:
            # Append to existing file
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['features'].extend(features)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)

        tiles_written[filename] = tiles_written.get(filename, 0) + len(features)

    except Exception as e:
        print(f"    ✗ Error writing to {filename}: {e}")


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
        output_tiles: List of output tile bounds
        output_dir: Directory to save output tiles
        tiles_written: Dictionary tracking feature counts per tile
    """
    print(f"\n  Processing {input_file.name}...")

    # Estimate feature count for progress reporting
    estimated_count = estimate_feature_count(input_file)
    if estimated_count > 0:
        print(f"  Estimated features: ~{estimated_count:,}")

    # Create output tile name lookup
    tile_files = {}
    for bounds in output_tiles:
        left_lon, upper_lat, right_lon, lower_lat = bounds
        filename = get_output_tile_name(left_lon, upper_lat, right_lon, lower_lat)
        tile_files[bounds] = output_dir / filename

    # Temporary storage for batched writes
    batch_buffers = {bounds: [] for bounds in output_tiles}
    features_processed = 0

    print(f"  Streaming features from file...")

    try:
        # Stream parse the JSON file
        with open(input_file, 'rb') as f:
            # Parse features array items one at a time
            # Use parse_float=float to avoid Decimal objects
            features = ijson.items(f, 'features.item', use_float=True)

            for feature in tqdm.tqdm(features, total=estimated_count):
                features_processed += 1

                # # Show progress every 1000 features
                # if features_processed % 1000 == 0:
                #     if estimated_count > 0:
                #         percent = min(100, (features_processed / estimated_count) * 100)
                #         print(f"\r  Processed: {features_processed:,} features (~{percent:.1f}%)", end='', flush=True)
                #     else:
                #         print(f"\r  Processed: {features_processed:,} features", end='', flush=True)

                # Check which tiles this feature belongs to
                for tile_bounds in output_tiles:
                    if feature_intersects_tile(feature, tile_bounds):
                        batch_buffers[tile_bounds].append(feature)

                # Flush batches when they get large enough
                for tile_bounds, batch in batch_buffers.items():
                    if len(batch) >= BATCH_SIZE:
                        output_path = tile_files[tile_bounds]
                        flush_batch_to_file(output_path, batch, tiles_written)
                        batch_buffers[tile_bounds] = []  # Clear batch

        print(f"\r  Processed: {features_processed} features (complete)")

        if estimated_count > 0:
            accuracy = (features_processed / estimated_count) * 100
            print(f"  Estimation accuracy: {accuracy:.1f}% (estimated {estimated_count:,}, actual {features_processed:,})")

        # Flush remaining batches
        print(f"  Writing remaining features...")
        for tile_bounds, batch in batch_buffers.items():
            if batch:
                output_path = tile_files[tile_bounds]
                flush_batch_to_file(output_path, batch, tiles_written)

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
    print(f"  Batch size: {BATCH_SIZE} features")
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