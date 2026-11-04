#!/usr/bin/env python3
"""Read-only OS service and NTP alignment check; never adjusts CLOCK_REALTIME."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clock_sync import main

if __name__ == '__main__':
    raise SystemExit(main())
