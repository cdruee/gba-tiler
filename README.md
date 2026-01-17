# GlobalBuildingAtlas Downloader and Tiler

A high-performance Python tool for downloading and processing 
GlobalBuildingAtlas (GBA) building footprint data. 
Downloads large GeoJSON files via rsync and splits them into smaller,
manageable tiles with configurable resolution.

### GlobalBuildingAtlas 
is is a dataset providing global and complete coverage of 
building polygons (GBA.Polygon), heights (GBA.Height) 
and Level of Detail 1 (LoD1) 3D building models (GBA.LoD1).

- The dataset is available under: 
[doi:10.14459/2025mp1782307](https://doi.org/10.14459/2025mp1782307)

- It can be viewed online in the 
[ GlobalBuildingAtlas web viewer](https://tubvsig-so2sat-vm1.srv.mwn.de/)

- Citation:\
  Zhu, X. X., Chen, S., Zhang, F., Shi, Y., and Wang, Y., 2025: 
  _GlobalBuildingAtlas: an open global and complete dataset of building polygons,
  heights and LoD1 3D models_, Earth Syst. Sci. Data, **17**, 6647–6668,
  [doi:10.5194/essd-17-6647-2025](https://doi.org/10.5194/essd-17-6647-2025). 

## Features

- **Flexible Area Selection**: Define areas by bounding box or country name
- **Configurable Parallel Processing**: Auto, sequential, or custom worker count
- **High Performance**: Streaming JSON parsing with spatial indexing (~70k features/sec)
- **Memory Efficient**: Processes multi-GB files without loading into RAM
- **Space Optimized**: Float precision limited to 1mm (3 decimal places)
- **Progress Monitoring**: Real-time progress tracking in both sequential and parallel modes
- **Standard GeoJSON**: Outputs include CRS metadata and tile bounding boxes
- **CLI Interface**: Professional command-line interface with comprehensive help
- **Python API**: Can be called programmatically from Python code

## Requirements

### Required Dependencies

```bash
pip install ijson requests gdal
```

- **Python 3.8+**
- **ijson**: Streaming JSON parser
- **requests**: HTTP downloads for country boundaries
- **GDAL/OGR**: Shapefile processing (country boundaries)
- **rsync**: System command-line tool

### Optional Dependencies

None required - all features work with base installation.

### System Requirements

- **rsync** must be installed and available in PATH
- Sufficient disk space for downloaded tiles and output
- Internet connection for downloading source data

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install gba-tiler
```

After installation, the command is available globally:
```bash
gba-tiler --help
```

### Option 2: Install from Source

#### 1. Clone the Repository

```bash
git clone https://gitlab.rlp.net/druee/gba-tiler.git
cd gba-tiler
```

#### 2. Install Python Dependencies

```bash
# Required
pip install ijson requests gdal

# Optional (for progress bars)
pip install tqdm
```

#### 3. Install rsync (if not already installed)

**Ubuntu/Debian:**
```bash
sudo apt-get install rsync python3-gdal
```

**macOS:**
```bash
brew install rsync gdal
pip install gdal==$(gdal-config --version)
```

**Fedora/RHEL:**
```bash
sudo dnf install rsync python3-gdal
```

## Usage

### Basic Usage

#### Using Bounding Box

```bash
python gba_tiler.py --bbox <lon_min> <lat_min> <lon_max> <lat_max>
```

Example:
```bash
# Process area covering parts of Germany
python gba_tiler.py --bbox 5.0 45.0 15.0 55.0
```

#### Using Country Name

```bash
python gba_tiler.py --country <country_name>
```

Example:
```bash
# Process all of Germany
python gba_tiler.py --country Germany

# Process France
python gba_tiler.py --country "France"
```

#### Using ISO Country Codes

```bash
# Using ISO 2-letter code
python gba_tiler.py --iso2 <code>

# Using ISO 3-letter code
python gba_tiler.py --iso3 <code>
```

Example:
```bash
# Process Germany using ISO codes
python gba_tiler.py --iso2 DE
python gba_tiler.py --iso3 DEU

# Process France using ISO codes
python gba_tiler.py --iso2 FR
python gba_tiler.py --iso3 FRA
```

### Version Information

```bash
# Show version
python gba_tiler.py --version
```

### Logging Options

By default, the script shows warnings and info messages. Use logging options to control verbosity:

```bash
# Verbose output (shows detailed progress)
python gba_tiler.py --country Germany --verbose

# Debug output (shows all debugging information)
python gba_tiler.py --country Germany --debug

# Quiet mode (errors only, faster parallel processing)
python gba_tiler.py --country Germany --quiet
```

### Processing Mode

Control the number of parallel workers with the `--parallel` (`-p`) option:

```bash
# Automatic parallel (default - uses CPU cores - 1)
python gba_tiler.py --country Germany

# Sequential processing (detailed per-file progress)
python gba_tiler.py --country Germany --parallel 1

# Custom worker count (e.g., 4 workers)
python gba_tiler.py --country Germany --parallel 4

# Legacy sequential flag (deprecated, use --parallel 1)
python gba_tiler.py --country Germany --sequential
```

**Processing Mode Comparison:**

| Mode | Workers | Speed | Progress Display |
|------|---------|-------|------------------|
| **Automatic** (`-p 0` or default) | CPU - 1 | Fast | Combined: `Processing: 23% [10MB] 45% [20MB]` |
| **Sequential** (`-p 1`) | 1 | Slower | Per-file: `Processing (file 1/8): 23% [10.5 MB]` |
| **Custom** (`-p 4`) | 4 | Configurable | Combined: `Processing: 23% [10MB] 45% [20MB]` |

**When to use each mode:**
- **Automatic** (default): Best for most use cases, maximizes performance
- **Sequential** (`-p 1`): When you need detailed per-file monitoring
- **Custom** (`-p N`): On resource-constrained systems or when fine-tuning performance

### Advanced Options

```bash
python gba_tiler.py --country Germany \
    --delta 0.05 \              # Tile size: 0.05° (~5.5km at equator)
    --batch-size 2000 \         # Write 2000 features per batch
    --output-dir my_tiles \     # Custom output directory
    --temp-dir my_temp \        # Custom temp directory
    --verbose                   # Show progress
```

### Complete Command-Line Reference

```
usage: gba_tiler.py [-h] [--version]
                    (--bbox LON_MIN LAT_MIN LON_MAX LAT_MAX | --country NAME | --iso2 CODE | --iso3 CODE)
                    [-v | --debug | -q] [-p N] [-1] [--delta DEGREES] [--batch-size N]
                    [--output-dir DIR] [--temp-dir DIR]

GlobalBuildingAtlas Downloader and Tiler

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  
  Area specification (mutually exclusive - choose one):
  --bbox LON_MIN LAT_MIN LON_MAX LAT_MAX
                        Bounding box: min_lon min_lat max_lon max_lat (in degrees)
  --country NAME        Country name (e.g., "Germany", "France")
  --iso2 CODE           Country ISO 2-letter code (e.g., "DE", "FR")
  --iso3 CODE           Country ISO 3-letter code (e.g., "DEU", "FRA")
  
  Logging (mutually exclusive):
  -v, --verbose         Enable verbose output (INFO level)
  --debug               Enable debug output (DEBUG level)
  -q, --quiet           Quiet mode - errors only (ERROR level)
  
  Processing mode:
  -p N, --parallel N    Number of parallel workers (default: 0)
                        0 = automatic (CPU cores - 1)
                        1 = sequential processing
                        2+ = specific worker count
  -1, --sequential      [DEPRECATED] Use --parallel 1 instead
  
  Optional parameters:
  --delta DEGREES       Tile size in degrees (default: 0.10)
  --batch-size N        Batch size for writing features (default: 1000)
  --output-dir DIR      Output directory (default: GBA_tiles)
  --temp-dir DIR        Temporary directory (default: GBA_temp)
```

### Programmatic Usage (Python API)

The `gba_tiler` can also be used programmatically from Python code:

```python
import logging
import gba_tiler

# Optional: Configure logging before importing
logging.basicConfig(level=logging.INFO)

# Process by country name (automatic parallel)
gba_tiler.main(country="Luxembourg")

# Process by ISO code with custom settings
gba_tiler.main(
    iso2="DE",
    delta=0.05,
    batch_size=2000,
    output_dir="germany_tiles",
    parallel=4  # Use 4 workers
)

# Process by bounding box (sequential mode)
gba_tiler.main(
    bbox=(11.3, 48.0, 11.8, 48.3),
    delta=0.01,
    output_dir="munich_tiles",
    parallel=1  # Sequential processing
)

# Automatic parallel (default)
gba_tiler.main(country="France")  # parallel=0 by default
```

**API Parameters:**
- `parallel`: Number of workers (0=auto, 1=sequential, 2+=custom)
- `sequential`: [DEPRECATED] Use `parallel=1` instead

**Parameters:**
- `bbox`: Tuple of `(lon_min, lat_min, lon_max, lat_max)` or `None`
- `country`: Country name string or `None`
- `iso2`: ISO 2-letter code or `None`
- `iso3`: ISO 3-letter code or `None`
- `delta`: Tile size in degrees (default: `0.10`)
- `batch_size`: Features per batch (default: `1000`)
- `output_dir`: Output directory (default: `'GBA_tiles'`)
- `temp_dir`: Temporary directory (default: `'GBA_temp'`)
- `sequential`: Process files sequentially instead of in parallel (default: `False`)

**Notes:**
- Exactly one of `bbox`, `country`, `iso2`, or `iso3` must be provided
- Logging level is inherited from the calling program
- If no logging is configured, `WARNING` level is used

## Output Format

### Directory Structure

```
GBA_tiles/
├── e00500_n4500_e00510_n4510_lod1.geojson
├── e00510_n4500_e00520_n4510_lod1.geojson
└── ...

GBA_temp/
├── e005_n50_e010_n45.geojson    # Downloaded source files
└── ...
```

### Output File Naming

Format: `<e|w><lon>_<n|s><lat>_<e|w><lon>_<n|s><lat>_lod1.geojson`

- **Longitude**: 5 digits (centidegrees), e.g., `e00500` = 5.00°E
- **Latitude**: 4 digits (centidegrees), e.g., `n4500` = 45.00°N
- **Example**: `e00567_n5621_e00578_n5637_lod1.geojson`
  - Left: 5.67°E, Upper: 56.21°N
  - Right: 5.78°E, Lower: 56.37°N

### GeoJSON Structure

```json
{
  "type": "FeatureCollection",
  "name": "e00567_n5621_e00578_n5637_lod1",
  "bbox": [556597.0, 5621521.0, 567729.0, 5637278.0],
  "crs": {
    "type": "name",
    "properties": {
      "name": "urn:ogc:def:crs:EPSG::3857"
    }
  },
  "features": [...]
}
```

- **name**: Output tile name
- **bbox**: Tile bounding box in EPSG:3857 (Web Mercator meters)
- **crs**: Coordinate Reference System (Web Mercator)
- **features**: Building footprint features

### Coordinate Systems

- **Input CRS**: EPSG:3857 (Web Mercator) - coordinates in meters
- **Tile Grid**: WGS84 (EPSG:4326) - tile boundaries in degrees
- **Output CRS**: EPSG:3857 (Web Mercator) - preserved from input

## Performance

### Benchmarks

- **Processing Speed**: ~70,000 features/second
- **Spatial Index**: Checks ~1-10 tiles per feature (vs 10,000 without index)
- **Memory Usage**: O(BATCH_SIZE × num_tiles) - typically <1GB RAM
- **Example**: 23 million features processed in ~10 hours

### Optimization Tips

1. **Increase batch size** for faster I/O (use powers of 2: 1000, 2000, 4000)
2. **Larger tiles** (higher DELTA) = fewer files, faster processing
3. **SSD storage** significantly improves performance
4. **Disable tqdm** if running in background scripts

### Performance Characteristics

| Area Size | Tiles (0.1°) | Est. Time | Memory |
|-----------|--------------|-----------|--------|
| 1° × 1°   | 100          | ~1 hour   | <500MB |
| 5° × 5°   | 2,500        | ~5 hours  | <1GB   |
| 10° × 10° | 10,000       | ~10 hours | <2GB   |

## Technical Details

### Algorithm Overview

1. **Download**: Fetch 5°×5° source tiles via rsync from GBA server
2. **Stream Parse**: Use ijson to process features one-by-one (no full file load)
3. **Spatial Index**: 100km grid cells to quickly find candidate tiles
4. **Bbox Center**: Calculate bbox center for each building
5. **Tile Assignment**: Assign building to tile containing its bbox center
6. **Batch Write**: Write features in batches for optimal I/O

### Coordinate Handling

Buildings are assigned to tiles based on their **bounding box center**:

```python
center_x = (min_x + max_x) / 2.0
center_y = (min_y + max_y) / 2.0
```

This ensures:
- ✅ Each building appears in exactly **one tile**
- ✅ No duplicate buildings across tiles
- ✅ Deterministic assignment (repeatable)

### Memory Management

The script uses **streaming** to handle large files:

- Input files are **never loaded** fully into memory
- Features are processed **one at a time** with ijson
- Output is written in **batches** (default: 1000 features)
- Batch buffers are **flushed** regularly to disk

### Floating Point Precision

All coordinates are rounded to **3 decimal places** (1mm in Web Mercator):

- Reduces file size by ~50%
- 0.001m precision is more than sufficient for buildings
- Maintains visual quality while saving disk space

## Example Scripts

### Basic API Usage

The repository includes `example_api.py` demonstrating programmatic usage:

```python
#!/usr/bin/env python3
"""Example: Using gba_tiler programmatically"""
import logging
import gba_tiler

# Configure logging before using gba_tiler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Example 1: Process by country name with defaults
gba_tiler.main(country="Luxembourg")

# Example 2: Process by ISO code with custom settings
gba_tiler.main(
    iso2="DE",
    delta=0.05,  # Smaller tiles
    batch_size=2000,  # Larger batches
    output_dir="germany_tiles"
)

# Example 3: Process by bounding box
gba_tiler.main(
    bbox=(11.3, 48.0, 11.8, 48.3),
    delta=0.01,  # Very small tiles for detail
    output_dir="munich_tiles"
)
```

Run the example:
```bash
python example_api.py
```

See [example_api.py](example_api.py) for complete examples including different logging levels.

### Batch Processing Multiple Countries

The repository includes `example_EU_countries.py` for batch processing:

```python
#!/usr/bin/env python3
"""Example: Batch processing EU countries"""
import csv
import lzma
import tarfile
from pathlib import Path
import gba_tiler

# Process multiple countries from CSV
with open('countries_EU.csv', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        country = row['Country']
        iso2 = row['ISO-Code']
        
        print(f"Processing: {country} ({iso2})")
        
        # Create tiles
        outdir = Path(f"GBA_tiles/{iso2.lower()}")
        gba_tiler.main(iso2=iso2, output_dir=str(outdir))
        
        # Compress to tar.xz
        with lzma.open(f"GBA_tiles.{iso2.lower()}.tar.xz", 'wb') as xz:
            with tarfile.open(fileobj=xz, mode='w') as tar:
                tar.add(outdir, arcname=iso2.lower())
```

This example processes all EU member states plus UK, Norway, Switzerland, and former Yugoslavia countries.

Run the batch processor:
```bash
python example_EU_countries.py
```

The script reads from [countries_EU.csv](countries_EU.csv) and creates compressed archives for each country.

## Configuration

### Default Values

```python
DELTA = 0.10          # Tile size in degrees (~11km at equator)
BATCH_SIZE = 1000     # Features per write batch
LAT_MIN = 45.0        # Minimum latitude (if using bbox)
LAT_MAX = 55.0        # Maximum latitude
LON_MIN = 5.0         # Minimum longitude
LON_MAX = 15.0        # Maximum longitude
OUTPUT_DIR = "GBA_tiles"
TEMP_DIR = "GBA_temp"
```

### Tile Size Guidelines

| DELTA | Tile Size (equator) | Files (10°×10°) | Use Case |
|-------|---------------------|-----------------|----------|
| 0.25° | ~28km × 28km        | 1,600           | Country-level analysis |
| 0.10° | ~11km × 11km        | 10,000          | **Default - balanced** |
| 0.05° | ~5.5km × 5.5km      | 40,000          | City-level detail |
| 0.01° | ~1.1km × 1.1km      | 1,000,000       | Neighborhood detail |

## Data Source

### GlobalBuildingAtlas

- **Source**: Technische Universität München (TUM)
- **Coverage**: Global building footprints
- **Resolution**: LoD1 (Level of Detail 1)
- **Format**: GeoJSON with EPSG:3857 coordinates
- **Access**: see https://mediatum.ub.tum.de/1782307

- rsync Server
  This program loads the data from the rsync server at the TUM library
  (Universitätsbibliothek der Technischen Universität München)
  using the (publicly available) credentials:

  ```
  rsync://m1782307:m1782307@dataserv.ub.tum.de/m1782307/LoD1/europe/
  ```

  **Note**: Update credentials in the script if needed.

### Country Boundaries

- **Source**: Natural Earth Data (public domain)
- **URL**: https://naciscdn.org/naturalearth/
- **Resolutions**: 110m (simplified), 10m (detailed)
- **Format**: Shapefile (auto-converted to GeoJSON)

## Troubleshooting

### Common Issues

#### 1. "rsync command not found"

Install rsync:
```bash
# Ubuntu/Debian
sudo apt-get install rsync

# macOS
brew install rsync
```

#### 2. "GDAL/OGR module is required"

Install GDAL:
```bash
# Ubuntu/Debian
sudo apt-get install python3-gdal

# macOS
brew install gdal
pip install gdal==$(gdal-config --version)

# pip (may require compilation)
pip install gdal
```

#### 3. "Country 'XYZ' not found"

- Check spelling (case-insensitive search)
- Try alternative names: "United States", "USA", "America"
- Use `--debug` to see available countries
- Use `--bbox` as fallback

#### 4. Memory Issues

- Increase system swap space
- Reduce `--batch-size` (default: 1000)
- Process smaller areas
- Use larger `--delta` (fewer tiles)

#### 5. No Output Files Created

- Check if input area has any buildings
- Verify rsync credentials
- Use `--debug` to see detailed processing info
- Check that tile boundaries overlap with data

### Performance Issues

**Slow Processing:**
- Increase `--batch-size` (try 2000-5000)
- Use SSD storage
- Disable antivirus scanning on work directories
- Increase `--delta` for fewer tiles

**High Memory Usage:**
- Decrease `--batch-size`
- Process smaller areas
- Close other applications

## Examples

### Example 1: Process Germany

```bash
# Using country name
python gba_tiler.py --country Germany --verbose

# Using ISO-2 code
python gba_tiler.py --iso2 DE --verbose

# Using ISO-3 code
python gba_tiler.py --iso3 DEU --verbose
```

### Example 2: Custom Tile Size

```bash
# Larger tiles for overview
python gba_tiler.py --country France --delta 0.25 --verbose

# Smaller tiles for detail
python gba_tiler.py --bbox 8.5 50.0 8.8 50.2 --delta 0.05 --debug
```

### Example 3: High Performance Configuration

```bash
python gba_tiler.py --country Germany \
    --batch-size 5000 \
    --verbose
```

### Example 4: Custom Directories

```bash
python gba_tiler.py --country Germany \
    --output-dir ~/data/germany_tiles \
    --temp-dir ~/data/temp \
    --verbose
```

### Example 5: Specific Region

```bash
# Berlin area
python gba_tiler.py --bbox 13.0 52.3 13.8 52.7 --verbose

# Munich area
python gba_tiler.py --bbox 11.3 48.0 11.8 48.3 --verbose
```

### Example 6: Version Information

```bash
# Check version
python gba_tiler.py --version
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes (follow PEP 8)
4. Add tests if applicable
5. Submit a pull request

## License

Copyright (C) 2025 Clemens Drüe, Universität Trier

This project is licensed under the 
[European Union Public License v1.2 (EUPL-1.2)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12).

See the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in research, please cite:

```
[Add citation information]
```

## Contact

- **Email**: druee@uni-trier.de


---

**Last Updated**: 07 Jan 2026