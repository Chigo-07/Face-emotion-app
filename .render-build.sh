#!/usr/bin/env bash

# Exit on errors
set -e

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Optional: collect static files (if using Flask with static assets)
# flask collect  # Uncomment if your app needs it

# Print Python version and packages for debugging
python --version
pip list

