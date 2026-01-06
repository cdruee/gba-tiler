#!/usr/bin/env python3
# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2025 Clemens Drüe, Universität Trier
"""
Test if GeoJSON Tiles Intersect with Country Borders

This script checks whether GeoJSON tile files intersect with the actual land surface
of a specified country by downloading the country border polygon from Natural Earth
and performing geometric intersection tests.

Features:
- Downloads country borders from Natural Earth Data
- Uses simplified polygon for fast pre-testing
- Uses detailed polygon for accurate intersection testing
- Supports any country by name
- Caches downloaded data for faster subsequent runs
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import math
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
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Error: requests module is required")
    print("Install with: pip install requests")
    sys.exit(1)

try:
    from osgeo import ogr, osr
    HAS_OGR = True
except ImportError:
    HAS_OGR = False
    print("Error: GDAL/OGR module is required")
    print("Install with: pip install gdal")
    print("Or on Ubuntu/Debian: sudo apt-get install python3-gdal")
    sys.exit(1)

# Configuration
INPUT_DIR = "GBA_tiles"
CACHE_DIR = ".country_borders_cache"
COUNTRY_NAME = "Germany"  # Change this to test other countries

# Natural Earth Data URLs - using multiple mirrors with fallback
# Note: naciscdn.org (not .com), and correct GitHub paths
DETAILED_URLS = [
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip",
    "https://github.com/nvkelso/natural-earth-vector/archive/refs/heads/master.zip",  # Full repo (fallback)
    "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip"
]

SIMPLIFIED_URLS = [
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
    "https://github.com/nvkelso/natural-earth-vector/archive/refs/heads/master.zip",  # Full repo (fallback)
    "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip"
]

# Earth radius for Web Mercator projection
EARTH_RADIUS = 6378137.0  # meters


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


def parse_filename(filename: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse tile boundaries from filename.
    
    Filename format: e00567_n5621_e00578_n5637_lod1.geojson
    
    Args:
        filename: Name of the GeoJSON file
    
    Returns:
        Tuple of (left_lon, upper_lat, right_lon, lower_lat) in degrees, or None if invalid
    """
    name = filename.replace('.geojson', '')
    pattern = r'^([ew])(\d{5})_([ns])(\d{4})_([ew])(\d{5})_([ns])(\d{4})(?:_lod1)?$'
    match = re.match(pattern, name)
    
    if not match:
        return None
    
    lon1_dir, lon1_val, lat1_dir, lat1_val, lon2_dir, lon2_val, lat2_dir, lat2_val = match.groups()
    
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


def get_tile_bbox_wgs84(filepath: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Get tile bounding box in WGS84 coordinates.
    
    First tries to read from the GeoJSON bbox field (if in Mercator, converts to WGS84).
    Falls back to parsing from filename.
    
    Args:
        filepath: Path to GeoJSON file
    
    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat) in WGS84 degrees, or None if error
    """
    # Try to read bbox from file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'bbox' in data:
            bbox = data['bbox']
            if len(bbox) == 4:
                # Check if bbox is in Mercator (large values) or WGS84 (small values)
                if abs(bbox[0]) > 180 or abs(bbox[1]) > 90:
                    # Mercator coordinates - convert to WGS84
                    min_x, min_y, max_x, max_y = bbox
                    min_lon, min_lat = mercator_to_wgs84(min_x, min_y)
                    max_lon, max_lat = mercator_to_wgs84(max_x, max_y)
                    return (min_lon, min_lat, max_lon, max_lat)
                else:
                    # Already WGS84
                    return tuple(bbox)
    except Exception:
        pass
    
    # Fall back to parsing filename
    bounds = parse_filename(filepath.name)
    if bounds:
        left_lon, upper_lat, right_lon, lower_lat = bounds
        return (left_lon, lower_lat, right_lon, upper_lat)
    
    return None


def download_and_extract_geojson(urls: List[str], cache_file: Path) -> Optional[Dict]:
    """
    Download Natural Earth shapefile ZIP and extract GeoJSON using OGR.
    Tries multiple URL sources with fallback.
    
    Args:
        urls: List of URLs to try (in order)
        cache_file: Path to cache the extracted GeoJSON
    
    Returns:
        GeoJSON FeatureCollection dict, or None if error
    """
    if cache_file.exists():
        print(f"  Using cached data: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    print(f"  Downloading from Natural Earth...")
    
    # Try each URL until one succeeds
    response = None
    successful_url = None
    
    for url in urls:
        print(f"  Trying: {url}")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            successful_url = url
            print(f"  ✓ Download successful")
            break
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    if response is None:
        print(f"  Error: All download sources failed")
        return None
    
    print(f"  Extracting shapefile...")
    
    # Extract the shapefile from ZIP
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Find the .shp file
            shp_files = [f for f in z.namelist() if f.endswith('.shp')]
            if not shp_files:
                print("  Error: No .shp file found in ZIP")
                return None
            
            shp_name = shp_files[0]
            base_name = shp_name.replace('.shp', '')
            
            # Extract all related files to temp directory
            temp_dir = Path(CACHE_DIR) / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                filename = base_name + ext
                if filename in z.namelist():
                    z.extract(filename, temp_dir)
            
            # Convert to GeoJSON using OGR
            shp_path = temp_dir / shp_name
            
            print(f"  Converting to GeoJSON using OGR...")
            
            # Open the shapefile with OGR
            driver = ogr.GetDriverByName('ESRI Shapefile')
            datasource = driver.Open(str(shp_path), 0)  # 0 means read-only
            
            if datasource is None:
                print(f"  Error: Could not open shapefile: {shp_path}")
                return None
            
            layer = datasource.GetLayer()
            
            # Convert to GeoJSON structure
            features = []
            for feature in layer:
                geom = feature.GetGeometryRef()
                if geom:
                    geom_json = json.loads(geom.ExportToJson())
                else:
                    geom_json = None
                
                # Get attributes
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
            
            datasource = None  # Close the datasource
            
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }
            
            # Cache the result
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(geojson, f)
            
            print(f"  Cached to: {cache_file}")
            return geojson
            
    except Exception as e:
        print(f"  Error extracting: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_country_geometry(country_name: str, detailed: bool = True):
    """
    Load country geometry from Natural Earth data.
    
    Args:
        country_name: Name of the country (e.g., "Germany", "France")
        detailed: If True, use 10m resolution; if False, use 110m (simplified)
    
    Returns:
        OGR Geometry object, or None if not found
    """
    resolution = "10m" if detailed else "110m"
    cache_file = Path(CACHE_DIR) / f"countries_{resolution}.geojson"
    urls = DETAILED_URLS if detailed else SIMPLIFIED_URLS
    
    # Download and extract
    geojson = download_and_extract_geojson(urls, cache_file)
    if not geojson:
        return None
    
    # Find the country
    for feature in geojson.get('features', []):
        properties = feature.get('properties', {})
        
        # Try multiple name fields
        names = [
            properties.get('NAME', ''),
            properties.get('NAME_LONG', ''),
            properties.get('ADMIN', ''),
            properties.get('NAME_EN', '')
        ]
        
        if any(country_name.lower() in name.lower() for name in names):
            print(f"  Found: {properties.get('NAME', 'Unknown')}")
            
            # Convert GeoJSON geometry to OGR geometry
            geom_json = json.dumps(feature['geometry'])
            geom = ogr.CreateGeometryFromJson(geom_json)
            
            return geom
    
    print(f"  Error: Country '{country_name}' not found")
    print(f"  Available countries in dataset:")
    for feature in geojson.get('features', [])[:10]:
        print(f"    - {feature['properties'].get('NAME', 'Unknown')}")
    print(f"    ... and {len(geojson['features']) - 10} more")
    return None


def get_country_bbox(geometry) -> Tuple[float, float, float, float]:
    """
    Get the bounding box of a country geometry.
    
    Args:
        geometry: OGR Geometry object
    
    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)
    """
    envelope = geometry.GetEnvelope()
    # OGR envelope is (minX, maxX, minY, maxY)
    return (envelope[0], envelope[2], envelope[1], envelope[3])  # Convert to (minX, minY, maxX, maxY)


def test_tile_intersection(tile_bbox: Tuple[float, float, float, float],
                          country_geom_simple,
                          country_geom_detailed,
                          country_bbox: Tuple[float, float, float, float]) -> str:
    """
    Test if a tile intersects with a country.
    
    Uses a two-stage approach:
    1. Fast bbox check against country bbox
    2. Fast pre-test with simplified country geometry
    3. Accurate test with detailed country geometry
    
    Args:
        tile_bbox: Tile bounding box (min_lon, min_lat, max_lon, max_lat)
        country_geom_simple: Simplified country geometry (OGR Geometry)
        country_geom_detailed: Detailed country geometry (OGR Geometry)
        country_bbox: Country bounding box (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        "outside" if no intersection
        "intersects" if intersects
    """
    tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat = tile_bbox
    country_min_lon, country_min_lat, country_max_lon, country_max_lat = country_bbox
    
    # Stage 1: Fast bbox check
    if (tile_max_lon < country_min_lon or tile_min_lon > country_max_lon or
        tile_max_lat < country_min_lat or tile_min_lat > country_max_lat):
        return "outside"
    
    # Create tile geometry as OGR Polygon
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(tile_min_lon, tile_min_lat)
    ring.AddPoint(tile_max_lon, tile_min_lat)
    ring.AddPoint(tile_max_lon, tile_max_lat)
    ring.AddPoint(tile_min_lon, tile_max_lat)
    ring.AddPoint(tile_min_lon, tile_min_lat)  # Close the ring
    
    tile_geom = ogr.Geometry(ogr.wkbPolygon)
    tile_geom.AddGeometry(ring)
    
    # Stage 2: Fast pre-test with simplified geometry
    if not country_geom_simple.Intersects(tile_geom):
        return "outside"
    
    # Stage 3: Accurate test with detailed geometry
    if country_geom_detailed.Intersects(tile_geom):
        return "intersects"
    
    return "outside"


def main():
    """Main execution function."""
    print(f"GeoJSON Tile - Country Intersection Tester v{__version__}")
    print("=" * 60)
    print(f"Country: {COUNTRY_NAME}")
    print(f"Input directory: {INPUT_DIR}")
    print()
    
    # Create cache directory
    Path(CACHE_DIR).mkdir(exist_ok=True)
    
    # Load country geometries
    print("Loading country border data...")
    print()
    
    print("Loading simplified geometry (110m)...")
    country_geom_simple = load_country_geometry(COUNTRY_NAME, detailed=False)
    if not country_geom_simple:
        sys.exit(1)
    
    print()
    print("Loading detailed geometry (10m)...")
    country_geom_detailed = load_country_geometry(COUNTRY_NAME, detailed=True)
    if not country_geom_detailed:
        sys.exit(1)
    
    # OGR geometries don't need "preparation" like shapely - they're ready to use
    country_geom_simple_prep = country_geom_simple
    country_geom_detailed_prep = country_geom_detailed
    
    # Get country bounding box
    country_bbox = get_country_bbox(country_geom_detailed)
    print()
    print(f"Country bounding box:")
    print(f"  Longitude: {country_bbox[0]:.3f}° to {country_bbox[2]:.3f}°")
    print(f"  Latitude: {country_bbox[1]:.3f}° to {country_bbox[3]:.3f}°")
    print()
    
    # Find all GeoJSON files
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists():
        print(f"Error: Input directory '{INPUT_DIR}' does not exist")
        sys.exit(1)
    
    geojson_files = list(input_dir.glob("*.geojson"))
    if not geojson_files:
        print(f"No GeoJSON files found in {INPUT_DIR}")
        sys.exit(0)
    
    print(f"Found {len(geojson_files)} GeoJSON file(s)")
    print()
    print("Testing intersections...")
    print()
    
    # Test each file
    intersecting_files = []
    outside_files = []
    error_files = []
    
    for idx, filepath in enumerate(geojson_files, 1):
        if idx % 100 == 0 or idx == len(geojson_files):
            print(f"  Progress: {idx}/{len(geojson_files)}", end='\r')
        
        # Get tile bbox
        tile_bbox = get_tile_bbox_wgs84(filepath)
        if not tile_bbox:
            error_files.append(filepath.name)
            continue
        
        # Test intersection
        result = test_tile_intersection(
            tile_bbox,
            country_geom_simple_prep,
            country_geom_detailed_prep,
            country_bbox
        )
        
        if result == "intersects":
            intersecting_files.append(filepath.name)
        elif result == "outside":
            outside_files.append(filepath.name)
    
    print()
    print()
    
    # Summary
    print("=" * 60)
    print("Results:")
    print(f"  Total files: {len(geojson_files)}")
    print(f"  Intersecting with {COUNTRY_NAME}: {len(intersecting_files)}")
    print(f"  Outside {COUNTRY_NAME}: {len(outside_files)}")
    if error_files:
        print(f"  Errors: {len(error_files)}")
    print()
    
    # Show intersecting files
    if intersecting_files:
        print(f"Files intersecting with {COUNTRY_NAME}:")
        for filename in intersecting_files[:20]:
            print(f"  - {filename}")
        if len(intersecting_files) > 20:
            print(f"  ... and {len(intersecting_files) - 20} more")
        print()
        
        # Save to file
        output_file = f"tiles_in_{COUNTRY_NAME.lower().replace(' ', '_')}.txt"
        with open(output_file, 'w') as f:
            for filename in intersecting_files:
                f.write(filename + '\n')
        print(f"Full list saved to: {output_file}")


if __name__ == "__main__":
    main()
            