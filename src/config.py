"""
Configuration file for paths and constants
"""
import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
OUTPUT_VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
OUTPUT_REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

# Models directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
for dir_path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUT_DIR, 
                 OUTPUT_VISUALIZATIONS_DIR, OUTPUT_REPORTS_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

