"""Build the ensemble app into dist/ensemble.

Downloads daily ICOS FLUXNET data, keeps the sites with at least 95% of every
variable in a year, and writes a static page. Needs ``requirements.txt``.

    python scripts/build_ensemble.py
"""
import base64
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from fluxscale.extra.midi import PROGRAMS
from fluxscale.extra.sonification import INSTRUMENTS

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "ensemble"
OUT = ROOT / "dist" / "ensemble"
STORE = "https://zarr.icos-cp.eu/icos-fluxnet.zarr"
YEARS = [2022, 2023, 2024, 2025]
COMPLETE = 0.95
VARIABLES = [  # store name, short name, label, family
    ("NEE_VUT_REF", "NEE", "Net ecosystem exchange", "carbon"),
    ("GPP_NT_VUT_REF", "GPP", "Gross primary production", "carbon"),
    ("RECO_NT_VUT_REF", "Reco", "Ecosystem respiration", "carbon"),
    ("LE_F_MDS", "LE", "Latent heat flux", "water"),
    ("VPD_F", "VPD", "Vapour pressure deficit", "water"),
    ("P_F", "Rain", "Precipitation", "weather"),
    ("H_F_MDS", "H", "Sensible heat flux", "energy"),
    ("SW_IN_F", "Sun", "Incoming shortwave radiation", "energy"),
    ("TA_F", "Temp", "Air temperature", "energy"),
    ("WS_F", "Wind", "Wind speed", "weather"),
]


def pack(values, offset, scale):
    """Values as 16-bit integers around zero; -32768 marks a missing day."""
    packed = np.full(values.shape, -32768, dtype="<i2")
    known = np.isfinite(values)
    packed[known] = np.round((values[known] - offset) / scale - 32000).astype("<i2")
    return packed


def main():
    names = [v[0] for v in VARIABLES]
    store = xr.open_zarr(STORE, group="_combined/fluxnet_dd", consolidated=True)
    daily = store[names].sel(time=slice(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31")).load()
    times = pd.DatetimeIndex(daily.time.values)
    stations = [str(s) for s in daily.station.values]
    known = np.stack([np.isfinite(daily[n].values) for n in names])

    offset = [float(np.nanmin(daily[n].values)) for n in names]
    scale = [float((np.nanmax(daily[n].values) - low) / 64000) or 1.0 for n, low in zip(names, offset)]

    (OUT / "data").mkdir(parents=True, exist_ok=True)
    years_of = {}
    packed_years = {}
    for year in YEARS:
        mask = times.year == year
        sites = {}
        complete = known[:, :, mask].mean(axis=2).min(axis=0) >= COMPLETE
        for i, station in enumerate(stations):
            if not complete[i]:
                continue
            table = np.stack([pack(daily[n].values[i, mask], offset[j], scale[j]) for j, n in enumerate(names)])
            sites[station] = base64.b64encode(table.tobytes()).decode()
            years_of.setdefault(station, []).append(year)
        packed_years[year] = dict(year=year, days=int(mask.sum()), sites=sites)
        print(year, len(sites), "sites")

    meta = dict(
        years=YEARS, offset=offset, scale=scale,
        vars=[dict(id=n, short=short, label=label, family=family, unit=daily[n].attrs.get("units"))
              for n, short, label, family in VARIABLES],
        sites={s: dict(name=str(store.station_name.values[i]), lat=round(float(store.latitude.values[i]), 3),
                       lon=round(float(store.longitude.values[i]), 3), eco=str(store.ecosystem.values[i]))
               for i, s in enumerate(stations) if s in years_of})
    timbres = {name: dict(partials=[list(p) for p in t["partials"]], attack=t["attack"], release=t["release"],
                          ring=t["ring"], vibrato=t["vibrato"], program=PROGRAMS[name])
               for name, t in INSTRUMENTS.items()}

    for year in YEARS[1:]:
        (OUT / "data" / f"{year}.json").write_text(json.dumps(packed_years[year], separators=(",", ":")))
    page = (APP / "template.html").read_text(encoding="utf-8")
    fill = {"@@BASE_CSS@@": (ROOT / "apps" / "shared" / "base.css").read_text(encoding="utf-8"),
            "@@META@@": json.dumps(meta, separators=(",", ":")),
            "@@TIMBRES@@": json.dumps(timbres, separators=(",", ":")),
            "@@LAND@@": (APP / "land.json").read_text(encoding="utf-8"),
            "@@YEAR@@": json.dumps(packed_years[YEARS[0]], separators=(",", ":"))}
    for key, value in fill.items():
        page = page.replace(key, value)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print("wrote", OUT / "index.html")


if __name__ == "__main__":
    main()
