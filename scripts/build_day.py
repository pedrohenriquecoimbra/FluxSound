"""Render the whole day of w*co2 for the decomposition app.

Needs the 20 Hz level-6 files of the FR-Gri sample, which live in the fluxscale
repository and are too big to keep here, so the results are kept instead:
``apps/decomposition/day.mp3`` and ``day.json``.

    python scripts/build_day.py PATH/TO/eddypro_raw_datasets/level_6
"""
import glob
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile
import xarray as xr
from fluxscale.extra import eddypro  # noqa: F401  (registers the level-6 reader)
from fluxscale.extra import sonification as so
from fluxscale.io import READERS

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "decomposition"
DAY = "20220513"
SECONDS = 60.0
WINDOW = 1800.0  # s of data: the eddy-covariance half hour
FS = 20.0


def main(folder):
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    files = sorted(f for f in glob.glob(os.path.join(folder, "*")) if os.path.basename(f).startswith(DAY))
    data = xr.concat([READERS.get("ep_raw_lvl")(f) for f in files], dim="TIMESTAMP")
    hours = len(files) / 2  # half-hour blocks, starting at 00:30

    audio = so.sonify(data, "w*co2", duration=SECONDS, fs=FS, correlation_window=WINDOW)
    soundfile.write(APP / "day.mp3", audio.values, so.SAMPLE_RATE, format="MP3")

    source = so._load(data, "w*co2", FS, "db6")
    score = so._score(source, SECONDS, WINDOW)
    n = score["correlation"].size
    at = np.linspace(0, n - 1, 300).astype(int)
    curve = dict(t=np.round(at / (n - 1) * SECONDS, 2).tolist(), r=np.round(score["correlation"][at], 3).tolist(),
                 hour=np.round(0.5 + at / (n - 1) * hours, 2).tolist())

    root, ratios = 65.41, so.SCALES["pentatonic"]
    low = so._scale_notes(2 * root, ratios, 8 * root)
    high = so._scale_notes(4 * root, ratios, 16 * root)
    slots, grid = so._note_grid(source["values"].size, n, SECONDS)
    lead, follower = so._duet(score, grid, low, high)
    height = np.array([so._harmonic_height(low[a], high[b]) for a, b in zip(lead, follower)])
    clock = 0.5 + slots / SECONDS * hours

    table = []
    for start in range(0, 24, 3):
        here = (clock >= start) & (clock < start + 3)
        a, b = np.sign(np.diff(lead[here])), np.sign(np.diff(follower[here]))
        both = (a != 0) & (b != 0)
        table.append(dict(hours=f"{start:02d}-{start + 3:02d}", r=round(float(score["correlation"][grid][here].mean()), 2),
                          consonant=round(float((height[here] <= so.CONSONANT_HEIGHT).mean()), 2),
                          contrary=round(float((a[both] != b[both]).mean()), 2) if both.any() else 0.0,
                          moves=int(both.sum()), direction=round(float(score["direction"][grid][here].mean()), 2)))
        print(table[-1])
    (APP / "day.json").write_text(json.dumps(dict(curve, table=table), separators=(",", ":")))
    print("r from", min(curve["r"]), "to", max(curve["r"]))


if __name__ == "__main__":
    main(sys.argv[1])
