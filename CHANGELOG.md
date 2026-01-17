# Changelog


## [1.6.0] - 2025-01-17

### Added:

- Configurable parallel processing with `-p/--parallel N` option
  - `0`: Automatic (CPU cores - 1, default)
  - `1`: Sequential processing
  - `2+`: Custom worker count
- API parameter `parallel` for programmatic control

### Changed:

- Updated all documentation to reflect new parallel options

### Deprecated:

- `-1/--sequential` flag (use `-p 1` or `--parallel 1` instead)
- `sequential` API parameter (use `parallel=1` instead)
- Both deprecated options still work but emit deprecation warnings

### Fixed:

- Example script logging configuration (now uses basicConfig before import)


## [1.5.1]  - 2026-01-16

### Fixed:

- updated documentation

## [1.5.0]  - 2026-01-16

### Added:

- option -1/--sequential to process sequentially
- option -q/--quiet to reduce logging output

### Changed:

- Parallel processing by default

## [1.4.1]  - 2026-01-11

### Fixed: 

- downloading way too much data to cover overseas territories

## [1.4.0]  - 2025-01-08

### Added:

- API 
- Programming examples for API use

### Changed:

- Updated documentation


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
