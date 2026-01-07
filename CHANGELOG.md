# Changelog

## [1.3.0]  - 2025-01-07

### Added:

- ISO country code support (--iso2, --iso3)
- Version display with --version
- Better error handling for JSON parsing
- File validation

### Changed:

- Improved country filtering logic
- Updated documentation

### Fixed:

- CI/CD issues (shallow clone, protected variables)
- Version file creation errors

## [1.2.0] - 2025-01-05

### Added:

- CLI with argparse
- Country boundary support
- Logging system
- Convenience script test_country_intersection.py
- CI/CD integration

## [1.1.0] - 2025-12-29

### Added:

- add_bbox_to_tiles.py script to annotate preexisting tiles

### Changed:

- Streaming JSON parsing
- Replaced shapely with GDAL/OGR
- Memory optimizations
- Coordinate conversion

## [1.0.0] - 2025-12-29

- Initial release
- Basic functionality
