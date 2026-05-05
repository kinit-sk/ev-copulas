"""Configuration module for the evcopulas project.

This module defines project-wide configuration settings including:

- Project directory structure and paths
- Data directory organization (raw, interim, processed, external)
- Model and report output directories
- Logging configuration with tqdm integration
"""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Project root and directory structure
PROJ_ROOT = Path(__file__).resolve().parents[1]

logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Data directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model and output directories
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure all directories exist
for _dir in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Logging configuration with tqdm integration
# If tqdm is installed, configure loguru with tqdm.write
# This prevents progress bars from being disrupted by log messages
# Reference: https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    # tqdm not available, use default loguru configuration
    pass
