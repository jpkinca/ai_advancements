"""Wrapper script to launch the extended MVP training located in src/train_mvp.py.
Allows running:
    python train_mvp.py [args]
from the repository root (tensortrade/) instead of cd into src.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
SRC_DIR = THIS_DIR / "src"
SRC_SCRIPT = SRC_DIR / "train_mvp.py"

if not SRC_SCRIPT.exists():
    raise SystemExit(f"Expected training script at {SRC_SCRIPT} not found.")

# Ensure src directory is on sys.path for module-relative imports inside the script.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Adjust sys.argv is fine; runpy will use current argv.
runpy.run_path(str(SRC_SCRIPT), run_name="__main__")
