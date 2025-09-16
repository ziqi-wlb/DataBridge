#!/usr/bin/env python3
"""
Setup script for DataBridge
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="databridge",
    version="0.1.0",
    author="ZiqiYu",
    author_email="550461236@qq.com",
    description="A comprehensive dataset conversion toolkit for transforming between different dataset formats",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DataBridge",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/DataBridge/issues",
        "Documentation": "https://github.com/yourusername/DataBridge#readme",
        "Source Code": "https://github.com/yourusername/DataBridge",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "webdataset>=0.2.0",
        "transformers>=4.20.0",
        "torch>=1.12.0",
        "tqdm>=4.64.0",
        "numpy>=1.21.0",
        "click>=8.0.0",
        "rich>=12.0.0",
        "megatron-core>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "databridge=databridge.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 