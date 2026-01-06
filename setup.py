#!/usr/bin/env python3
# SPDX-License-Identifier: EUPL-1.2
# Copyright (C) 2025 Clemens Drüe, Universität Trier

from setuptools import setup, find_packages

setup(
    name="gba-tiler",
    use_scm_version={
        "version_scheme": "python-simplified-semver",
        "local_scheme": "no-local-version",
    },
    setup_requires=["setuptools_scm"],
    author="Clemens Drüe",
    author_email="druee@uni-trier.de",
    description="High-performance tool for downloading and tiling GlobalBuildingAtlas data",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://gitlab.rlp.net/druee/gba-tiler",
    project_urls={
        "Bug Tracker": "https://gitlab.rlp.net/druee/gba-tiler/-/issues",
        "Documentation": "https://gitlab.rlp.net/druee/gba-tiler/-/blob/main/README.md",
        "Source Code": "https://gitlab.rlp.net/druee/gba-tiler",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    py_modules=["gba_tiler", "add_bbox_to_tiles", "test_country_intersection"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ijson>=3.1.0",
        "requests>=2.25.0",
        "GDAL>=3.0.0",
    ],
    extras_require={
        "progress": ["tqdm>=4.50.0"],
        "dev": [
            "setuptools_scm",
            "build",
            "twine",
        ],
    },
    entry_points={
        "console_scripts": [
            "gba-tiler=gba_tiler:main",
            "gba-add-bbox=add_bbox_to_tiles:main",
            "gba-test-country=test_country_intersection:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
