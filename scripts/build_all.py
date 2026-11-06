"""Build both apps and the landing page into dist/, ready to serve as static files."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for script in ("build_ensemble.py", "build_decomposition.py"):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)
shutil.copy(ROOT / "apps" / "index.html", ROOT / "dist" / "index.html")
shutil.copy(ROOT / "apps" / "shared" / "base.css", ROOT / "dist" / "base.css")
