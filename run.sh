#!/bin/bash
# Script to run main.py with the virtual environment activated
cd "$(dirname "$0")"
source .venv/bin/activate
python main.py

