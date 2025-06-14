#!/usr/bin/env python
"""
Entry point script for merging features.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from iv_drl.feats.merge_feats import main

if __name__ == "__main__":
    main() 