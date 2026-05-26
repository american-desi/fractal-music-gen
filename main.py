#!/usr/bin/env python3
"""Fractal Music Generator - entry point.

Run with: python main.py
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from fractal_music_gen.main import main

if __name__ == "__main__":
    main()
