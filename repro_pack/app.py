import os
import sys
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPRO_ROOT = ROOT / "repro_pack"
DEMO_APP_DIR = REPRO_ROOT / "demo_app"
FGTS_CORE_DIR = REPRO_ROOT / "fgts_core"

os.chdir(str(REPRO_ROOT))

for p in [str(REPRO_ROOT), str(DEMO_APP_DIR), str(FGTS_CORE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

runpy.run_path(str(DEMO_APP_DIR / "app.py"), run_name="__main__")