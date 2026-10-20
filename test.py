#!/usr/bin/env python3
"""Evaluate a supplied checkpoint in RoboTwin (or run a zero-model preflight)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from wam_pwtr.evaluate import main

if __name__ == '__main__':
    raise SystemExit(main())
