#!/usr/bin/env python3
"""Setup script for AI Trading Advancements package."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            requirements = [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith("#")
            ]
    return requirements

setup(
    name="ai-trading-advancements",
    version="0.1.0",
    author="Trading AI Research Team",
    author_email="research@example.com",
    description="Cutting-edge AI algorithmic trading techniques and implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ai-trading-advancements",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
            "cupy-cuda12x>=12.0.0",
        ],
        "quantum": [
            "qiskit>=0.43.0",
            "pennylane>=0.31.0",
        ],
        "blockchain": [
            "web3>=6.0.0",
            "eth-account>=0.9.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-trading-test=src.testing.test_runner:main",
            "ai-trading-setup=src.setup.database_setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
    },
    zip_safe=False,
)
