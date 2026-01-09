#!/usr/bin/env python3
"""
Example: Using gba_tiler programmatically
"""
import logging

# Configure your own logging before importing gba_tiler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Now import and use gba_tiler
import gba_tiler

# Example 1: Process by country name with defaults
print("=" * 60)
print("Example 1: Process Luxembourg with defaults")
print("=" * 60)
gba_tiler.main(country="Luxembourg")

# Example 2: Process by ISO code with custom tile size
print("\n" + "=" * 60)
print("Example 2: Process Germany with custom settings")
print("=" * 60)
gba_tiler.main(
    iso2="DE",
    delta=0.05,  # Smaller tiles
    batch_size=2000,  # Larger batches
    output_dir="germany_tiles",
    temp_dir="germany_temp"
)

# Example 3: Process by bounding box
print("\n" + "=" * 60)
print("Example 3: Process Munich area by bbox")
print("=" * 60)
gba_tiler.main(
    bbox=(11.3, 48.0, 11.8, 48.3),
    delta=0.01,  # Very small tiles for detail
    output_dir="munich_tiles"
)

# Example 4: With different logging level
print("\n" + "=" * 60)
print("Example 4: Process with DEBUG logging")
print("=" * 60)
logging.getLogger().setLevel(logging.DEBUG)
gba_tiler.main(
    iso3="LUX",
    delta=0.20
)
