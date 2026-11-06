"""Build the decomposition app into dist/decomposition.

Decomposes the sample and renders one audio stem per scale and level with
fluxscale, and the L2 music of each signal as MIDI. Needs ``requirements.txt``.

    python scripts/build_decomposition.py
    python scripts/build_decomposition.py --page      # keep the audio, rebuild the page
    python scripts/build_decomposition.py --spectra   # as --page, and redraw the spectrograms
"""
import base64
import html
import io
import json
import logging
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile
from PIL import Image
from scipy import signal as spsignal

from fluxscale import decompose, export_midi, sonify

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "decomposition"
OUT = ROOT / "dist" / "decomposition"
FS = 20.0
SECONDS = 24.0
COLUMNS = 900
SAMPLE_RATE = 44100
HEARD_RANGE = (20.0, 16000.0)  # Hz, the fixed axis of the spectrograms
HEARD_ROWS = 120
HEARD_COLUMNS = 400
HEARD_DB = (-110.0, -30.0)     # the page decodes grey levels with the same numbers
SIGNALS = [  # variable, button, label, unit
    ("w", "w", "Vertical wind", "m/s"),
    ("co2", "CO₂", "CO₂", "ppm"),
    ("ts", "T", "Sonic temperature", "K"),
    ("h2o", "H₂O", "Water vapour", "mmol/mol"),
    ("w*co2", "w×CO₂", "CO₂ flux density", "ppm m/s"),
    ("w*ts", "w×T", "Heat flux density", "K m/s"),
]


def encode(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def blocks(series, statistic):
    edges = np.linspace(0, series.size, COLUMNS + 1).astype(int)
    return np.array([statistic(series[a:b]) for a, b in zip(edges[:-1], edges[1:])])


def extreme(chunk):
    """The value furthest from zero, so that a short burst survives being drawn."""
    return chunk[np.abs(chunk).argmax()]


def period_label(frequency):
    low, high = 1 / frequency, 2 / frequency  # a band spans frequency/2 to frequency
    return f"{low:.1f}–{high:.1f} s" if high < 10 else f"{low:.0f}–{high:.0f} s"


def spectrogram(audio):
    """Grey levels of *audio* on the fixed axis: time across, log frequency up."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    mono = mono[:int(SECONDS * SAMPLE_RATE)]
    hop = mono.size // HEARD_COLUMNS
    freqs, _, power = spsignal.spectrogram(mono, fs=SAMPLE_RATE, nperseg=4096, noverlap=4096 - hop,
                                           window="hann", scaling="spectrum")
    level = 10 * np.log10(power + 1e-14)
    rows = np.geomspace(*HEARD_RANGE, HEARD_ROWS)[::-1]
    on_axis = np.array([np.interp(rows, freqs, column) for column in level.T]).T
    columns = np.linspace(0, on_axis.shape[1] - 1, HEARD_COLUMNS)
    grid = np.array([np.interp(columns, np.arange(on_axis.shape[1]), row) for row in on_axis])
    low, high = HEARD_DB
    return np.clip((grid - low) / (high - low) * 255, 0, 255).astype("u1")


def write_spectra(signals):
    """One grey image per signal and level, its stems stacked, from the audio as played."""
    for item in signals:
        for manifest in item["stems"].values():
            pack = (OUT / manifest[0]["file"]).read_bytes()
            stacked = []
            for stem in manifest:
                audio, _ = soundfile.read(io.BytesIO(pack[stem["offset"]:stem["offset"] + stem["length"]]))
                stacked.append(spectrogram(audio))
            Image.fromarray(np.vstack(stacked), mode="L").save((OUT / manifest[0]["file"]).with_suffix(".png"), optimize=True)
        print("spectra", item["variable"], flush=True)


def write_stems(table, variable, slug, level):
    """Render the stems of one signal at one level; returns the manifest."""
    stems = sonify(table, variable, level=level, duration=SECONDS, fs=FS, stems=True, sample_rate=SAMPLE_RATE)
    path = OUT / "audio" / slug / f"L{level}.mp3"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    names = [str(name) for name in stems.stem.values]
    # One file per signal and level, sliced by the page: a few requests, not hundreds.
    with open(path, "wb") as pack:
        for k, name in enumerate(names):
            audio = stems.values[k]
            encoded = io.BytesIO()
            soundfile.write(encoded, audio[:, 0] if level == 0 else audio, SAMPLE_RATE, format="MP3")
            manifest.append(dict(name=name, file=str(path.relative_to(OUT)), offset=pack.tell(),
                                 length=encoded.tell(), row=None))
            pack.write(encoded.getvalue())
    return manifest, stems.attrs, names


def write_midi(table, variable):
    """The L2 music of one signal as MIDI, with the arguments of its stems; returns its path."""
    path = OUT / "midi" / (variable.replace("*", "-") + ".mid")
    export_midi(table, variable, path, duration=SECONDS, fs=FS)
    return str(path.relative_to(OUT))


def build_signal(table, variable, short, label, unit):
    split = decompose(table, variable, fs=FS)
    frequencies = split.frequency.values
    rows = []
    for k, frequency in enumerate(frequencies):
        values = split.scales.values[k]
        scale = np.quantile(np.abs(values), 0.995) or 1.0
        rows.append(dict(
            hz=float(frequency), period=period_label(frequency), share=float(split.share.values[k]),
            strip=encode(np.clip(np.round(blocks(values, extreme) / scale * 127), -127, 127).astype("i1")),
            mean=encode(blocks(values, np.mean).astype("<f4"))))
    slug = variable.replace("*", "-")
    stems = {}
    for level in (0, 1, 2):
        manifest, attrs, names = write_stems(table, variable, slug, level)
        heard = dict(zip(attrs.get("band_frequency_hz", []), attrs.get("voice_frequency_hz", [])))
        for stem in manifest:
            matches = [k for k, f in enumerate(frequencies) if stem["name"] == f"{f:g} Hz"]
            stem["row"] = matches[0] if matches else None
            stem["short"] = {"bass": "Bass", "trend": "Trend"}.get(stem["name"], stem["name"].split(": ")[-1])
            stem["label"] = {"bass": "Lowest voice: root for upward flux, minor for downward"}.get(
                stem["name"], "Melody played by " + next((i.split(": ")[1] for i in attrs.get("instruments", [])
                                                          if i.startswith(stem["short"] + ":")), "the instrument"))
        stems[level] = manifest
        for k, frequency in enumerate(frequencies):
            rows[k].setdefault("heard", {})[level] = heard.get(float(frequency), 0.0)
    return dict(variable=variable, short=short, label=label, unit=unit, rows=rows, stems=stems,
                speedup=float(split.time.values[-1] / SECONDS),
                signal=encode(blocks(split.signal.values, np.mean).astype("<f4")),
                trend=encode(blocks(split.trend.values, np.mean).astype("<f4")))


def day_table(day):
    """The day in three-hour steps: hours across, one row per measure."""
    def percent(value):
        return f"{round(100 * value)}%"

    def signed(value):
        return "0.00" if abs(value) < 0.005 else ("−" if value < 0 else "+") + f"{abs(value):.2f}"

    def bass(value):
        return "minor" if value < 0.25 else "major" if value > 0.75 else "mixed"

    steps = day["table"]
    marks = [' class="locked"' if step["consonant"] >= 0.99 else "" for step in steps]

    def row(label, cells):
        return f"<tr><th>{label}</th>" + "".join(f"<td{m}>{c}</td>" for m, c in zip(marks, cells)) + "</tr>"

    hours = "".join(f"<th{m}>{html.escape(step['hours']).replace('-', '–')}</th>" for m, step in zip(marks, steps))
    return (f"<thead><tr><th>Hours</th>{hours}</tr></thead><tbody>"
            + row("Mean r", [signed(step["r"]) for step in steps])
            + row("Consonant", [percent(step["consonant"]) for step in steps])
            + row("Opposite motion", [percent(step["contrary"]) if step["moves"] >= 10 else "–" for step in steps])
            + row("Bass", [bass(step["direction"]) for step in steps])
            + "</tbody>")


def main():
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    cached = ROOT / "dist" / "decomposition-model.json"
    sample = np.load(ROOT / "data" / "fr-gri-2022-05-13-1230.npz")
    table = {name: sample[name].astype(float) for name in sample.files}
    if "--page" in sys.argv or "--spectra" in sys.argv:
        signals = json.loads(cached.read_text(encoding="utf-8"))
    else:
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir(parents=True)
        signals = []
        for variable, short, label, unit in SIGNALS:
            signals.append(build_signal(table, variable, short, label, unit))
            print("built", variable, flush=True)
        cached.write_text(json.dumps(signals, ensure_ascii=False), encoding="utf-8")
    if "--page" not in sys.argv:
        write_spectra(signals)
    for item in signals:  # quick enough to redo every time, so --page can offer it too
        item["midi"] = write_midi(table, item["variable"])
    day = json.loads((APP / "day.json").read_text(encoding="utf-8"))
    shutil.copy(APP / "day.mp3", OUT / "audio" / "day.mp3")
    page = (APP / "template.html").read_text(encoding="utf-8")
    fill = {"@@BASE_CSS@@": (ROOT / "apps" / "shared" / "base.css").read_text(encoding="utf-8"),
            "@@MODEL@@": json.dumps(dict(seconds=SECONDS, heardRange=HEARD_RANGE, heardDb=HEARD_DB, heardRows=HEARD_ROWS,
                                         signals=signals), separators=(",", ":"), ensure_ascii=False),
            "@@DAY@@": json.dumps(dict(t=day["t"], r=day["r"], hour=day["hour"]), separators=(",", ":")),
            "@@DAY_TABLE@@": day_table(day)}
    for key, value in fill.items():
        page = page.replace(key, value)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print("wrote", OUT, round(size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
