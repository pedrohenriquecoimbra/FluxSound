"""Turn any variable into sound.

Three levels, each a step further from the samples and closer to the ear:

- **L0, audio file.** The signal written as audio and nothing more. By default
  each data sample is one audio sample.
- **L1, audible.** The same signal altered for human hearing. Playing a record
  faster is not enough: speed and pitch are tied together, so squeezing half an
  hour of 20 Hz data into 30 s leaves most of its variance below the threshold
  of hearing. The wavelet transform unties them. Each scale is expanded into the
  hearing range on its own, then equalised and weighted for equal loudness.
  Every sound is still a piece of the data, and fine eddies stay above large
  ones.
- **L2, music.** Each wavelet scale drives a sustained voice on a consonant
  scale, and the value of the variable plays a melody over it. A flux such as
  ``w*co2`` is a duet: one instrument per variable, each playing its own line.
  Their covariance is what is heard as harmony. Correlated variables move in
  parallel, anti-correlated ones in contrary motion, and the stronger the local
  correlation the more consonant their interval. The voices underneath follow
  the flux at each scale, and the lowest one names its direction: the root for
  upward flux, the relative minor for downward.

Gaps in the data are silent. Over a long record, calm passages stay softer than
turbulent ones but within reach of the ear (see ``loudness_range``). Scales,
instruments and room are fixed and seeded, so what moves in the sound is what
moved in the signal.
"""

# built-in modules
import logging
import os
from fractions import Fraction

# 3rd party modules
import numpy as np
import pandas as pd
import pywt
import xarray as xr
from scipy import signal as spsignal
from scipy.io import wavfile
from scipy.ndimage import (map_coordinates, maximum_filter1d, spline_filter1d,
                           uniform_filter1d)

# waveletec
from waveletec.core.wavelet_functions import formula_to_vars, prepare_signal, universal_wt

logger = logging.getLogger(__name__)

__all__ = ["sonify", "write_wav", "integrated_loudness", "SCALES", "INSTRUMENTS"]

SAMPLE_RATE = 44100
DEFAULT_DURATION = 30.0      # s, L1 and L2

# Pure (just-intonation) ratios rather than equal temperament: voices here are
# sustained for seconds, and pure intervals make their overtones coincide
# instead of beating. Five-note scales have no semitone, so any combination of
# notes is consonant, which is what makes the mapping safe for arbitrary data.
SCALES = {
    "pentatonic": (1, 9 / 8, 5 / 4, 3 / 2, 5 / 3),
    "minor_pentatonic": (1, 6 / 5, 4 / 3, 3 / 2, 9 / 5),
    "suspended": (1, 9 / 8, 4 / 3, 3 / 2, 16 / 9),
}



def _timbre(partials, attack, release, ring=None, vibrato=0.0, octave=0):
    """An instrument, as additive synthesis.

    Args:
        partials: ``(harmonic, weight, decay in s)`` for each partial.
        attack, release: Fades at each end of a note, in s.
        ring: Length of a note in s, for struck and plucked instruments. ``None``
            holds it until the next one.
        vibrato: Depth as a fraction of the frequency.
        octave: Octaves from the notes of the line, to sit in its natural range.
            Harmony does not depend on it, since an octave counts as a unison.
    """
    return dict(partials=partials, attack=attack, release=release, ring=ring,
                vibrato=vibrato, octave=octave)


# Sketches of real instruments: what sets them apart is which partials they have
# and how fast those die. Partials stay on whole numbers, which is true of tuned
# bars too, so that every instrument stays in tune with the sustained voices. A
# held and a struck instrument differ in envelope, the strongest cue for telling
# two simultaneous lines apart, so that is the pairing to reach for in a duet.
INSTRUMENTS = {
    # held
    "flute": _timbre(((1, 1.0, 6.0), (2, 0.25, 4.0), (3, 0.08, 3.0)), 0.08, 0.3, vibrato=0.004),
    "clarinet": _timbre(((1, 1.0, 8.0), (2, 0.05, 8.0), (3, 0.6, 8.0), (5, 0.35, 6.0),
                         (7, 0.18, 5.0), (9, 0.08, 4.0)), 0.05, 0.12),  # odd partials: hollow
    "oboe": _timbre(((1, 0.55, 8.0), (2, 1.0, 8.0), (3, 0.85, 7.0), (4, 0.5, 6.0), (5, 0.3, 5.0),
                     (6, 0.18, 4.0), (7, 0.1, 4.0)), 0.045, 0.1, vibrato=0.003),  # weak fundamental: nasal
    "bassoon": _timbre(((1, 0.7, 8.0), (2, 1.0, 8.0), (3, 0.7, 7.0), (4, 0.35, 6.0), (5, 0.25, 5.0),
                        (6, 0.12, 4.0)), 0.06, 0.12, vibrato=0.002, octave=-1),
    "saxophone": _timbre(((1, 1.0, 8.0), (2, 0.6, 8.0), (3, 0.5, 7.0), (4, 0.4, 6.0), (5, 0.25, 5.0),
                          (6, 0.18, 4.0), (7, 0.1, 3.0)), 0.05, 0.12, vibrato=0.005),
    "horn": _timbre(((1, 1.0, 8.0), (2, 0.7, 8.0), (3, 0.4, 7.0), (4, 0.22, 6.0), (5, 0.12, 5.0),
                     (6, 0.06, 4.0)), 0.09, 0.18, octave=-1),
    "trumpet": _timbre(((1, 0.8, 8.0), (2, 1.0, 8.0), (3, 0.9, 7.0), (4, 0.7, 6.0), (5, 0.5, 5.0),
                        (6, 0.35, 4.0), (7, 0.22, 4.0), (8, 0.15, 3.0)), 0.05, 0.1, vibrato=0.002),
    "trombone": _timbre(((1, 1.0, 8.0), (2, 0.85, 8.0), (3, 0.6, 7.0), (4, 0.4, 6.0), (5, 0.25, 5.0),
                         (6, 0.14, 4.0)), 0.07, 0.14, octave=-1),
    "violin": _timbre(((1, 0.9, 8.0), (2, 0.8, 8.0), (3, 0.75, 7.0), (4, 0.5, 6.0), (5, 0.45, 5.0),
                       (6, 0.3, 4.0), (7, 0.22, 3.0), (8, 0.15, 3.0)), 0.09, 0.2, vibrato=0.006),
    "viola": _timbre(((1, 1.0, 8.0), (2, 0.75, 8.0), (3, 0.5, 7.0), (4, 0.35, 6.0), (5, 0.22, 5.0),
                      (6, 0.14, 4.0)), 0.11, 0.22, vibrato=0.005),
    "cello": _timbre(((1, 1.0, 8.0), (2, 0.8, 8.0), (3, 0.55, 7.0), (4, 0.4, 6.0), (5, 0.28, 5.0),
                      (6, 0.18, 4.0), (7, 0.12, 3.0), (8, 0.08, 3.0)), 0.14, 0.25,
                     vibrato=0.006, octave=-1),
    "double_bass": _timbre(((1, 1.0, 8.0), (2, 0.7, 8.0), (3, 0.35, 7.0), (4, 0.18, 6.0),
                            (5, 0.1, 5.0)), 0.16, 0.3, vibrato=0.003, octave=-2),
    "voice": _timbre(((1, 1.0, 8.0), (2, 0.55, 8.0), (3, 0.4, 7.0), (4, 0.12, 6.0), (5, 0.06, 5.0)),
                     0.16, 0.3, vibrato=0.007),
    "organ": _timbre(((1, 1.0, 60.0), (2, 0.75, 60.0), (3, 0.25, 60.0), (4, 0.4, 60.0),
                      (6, 0.2, 60.0), (8, 0.22, 60.0)), 0.015, 0.06),  # no decay at all
    "accordion": _timbre(((1, 1.0, 30.0), (2, 0.7, 30.0), (3, 0.5, 30.0), (4, 0.35, 30.0),
                          (5, 0.2, 30.0)), 0.03, 0.08, vibrato=0.002),
    # struck and plucked
    "bell": _timbre(((1, 1.0, 0.9), (2, 0.3, 0.5), (3, 0.1, 0.25)), 0.006, 0.2, ring=2.2),
    "piano": _timbre(((1, 1.0, 1.5), (2, 0.55, 1.0), (3, 0.3, 0.7), (4, 0.18, 0.45), (5, 0.1, 0.3),
                      (6, 0.06, 0.2)), 0.004, 0.25, ring=2.0),
    "harp": _timbre(((1, 1.0, 1.3), (2, 0.35, 0.8), (3, 0.12, 0.5), (4, 0.05, 0.3)),
                    0.004, 0.2, ring=2.0),
    "harpsichord": _timbre(((1, 0.7, 0.8), (2, 0.9, 0.6), (3, 0.7, 0.45), (4, 0.55, 0.3), (5, 0.4, 0.2),
                            (6, 0.3, 0.15), (7, 0.2, 0.1), (8, 0.12, 0.08)), 0.002, 0.08, ring=1.2),
    "guitar": _timbre(((1, 1.0, 0.9), (2, 0.7, 0.6), (3, 0.5, 0.4), (4, 0.3, 0.3), (5, 0.2, 0.2),
                       (6, 0.12, 0.15)), 0.003, 0.15, ring=1.5),
    "electric_guitar": _timbre(((1, 0.8, 1.6), (2, 1.0, 1.4), (3, 0.8, 1.2), (4, 0.6, 0.9), (5, 0.45, 0.7),
                                (6, 0.3, 0.5), (7, 0.2, 0.4), (8, 0.12, 0.3)), 0.003, 0.1, ring=1.6),
    "bass_guitar": _timbre(((1, 1.0, 1.2), (2, 0.5, 0.8), (3, 0.25, 0.5), (4, 0.1, 0.3)),
                           0.004, 0.1, ring=1.2, octave=-2),
    "pizzicato": _timbre(((1, 1.0, 0.3), (2, 0.6, 0.2), (3, 0.35, 0.12), (4, 0.2, 0.08)),
                         0.003, 0.06, ring=0.6, octave=-1),
    "marimba": _timbre(((1, 1.0, 0.45), (4, 0.18, 0.1), (10, 0.04, 0.03)), 0.002, 0.08, ring=0.9),
    "vibraphone": _timbre(((1, 1.0, 1.8), (4, 0.22, 0.5), (10, 0.03, 0.12)), 0.003, 0.3, ring=2.4),
    "glockenspiel": _timbre(((1, 1.0, 1.0), (3, 0.25, 0.3), (6, 0.08, 0.1)), 0.001, 0.15,
                            ring=1.4, octave=1),
    "celesta": _timbre(((1, 1.0, 0.9), (2, 0.2, 0.4), (4, 0.08, 0.12)), 0.002, 0.2, ring=1.4, octave=1),
    "timpani": _timbre(((1, 1.0, 0.7), (2, 0.25, 0.3), (3, 0.1, 0.15)), 0.003, 0.2, ring=1.4, octave=-1),
    "kalimba": _timbre(((1, 1.0, 0.6), (6, 0.2, 0.04)), 0.002, 0.1, ring=1.0),
}

AUDIBLE_RANGE = (20, 20000)  # Hz
PEAK_CEILING = 10 ** (-1 / 20)  # -1 dBFS
SLOPE_KEPT = 0.2             # share of the spectral slope left by equalisation
MAX_BOOST_DB = 30            # equalisation never lifts a band further than this
LOUDNESS_WINDOW = 3.0        # s, the short-term window of BS.1770

# L1
HEARING_BAND = (100, 8000)   # Hz, where the scales of the data are expanded to
GRAIN_CYCLES = 4             # length of a grain, in cycles of its own band
GRAIN_OVERLAP = 4
GRAIN_CHUNK = 2 ** 22        # samples read at a time, to bound memory
MAX_MODULATION_HZ = 8        # below the ~15 Hz where fluctuation turns rough

# L2
CONTROL_RATE = 200           # Hz, audio time; rate of the loudness and pan curves
LOUDNESS_EXPONENT = 0.6      # Stevens: loudness grows as pressure ** 0.6
PAD_TOP = 1320               # Hz, highest sustained voice
BASS_TOP = 200               # Hz, below which thirds turn muddy: roots and fifths only
PAD_MOD_HZ = 3               # slower than L1: swells, not tremolo
FLUX_SLOPE_KEPT = 0.5        # more for a flux: the cospectrum is the point
NOTE_RATE = 3                # melody slots per second
MAX_HOLD = 2.5               # s, longest held note
RING_OUT = 3.0               # s after the record, for the last notes and the room
LINE_LEVEL = 0.7             # melody level relative to the pad
CORRELATION_SECONDS = 3.0    # s of sound over which local correlation is judged
# Measured on the FR-Gri sample with w against co2, h2o and ts: |r| over such a
# window is ~0.06 by chance (0.14 at the 90th percentile) and ~0.45 at midday.
HARMONY_ONSET, HARMONY_FULL = 0.15, 0.5
# Harmonic height separating thirds, fourths, fifths, sixths and octaves from
# seconds and sevenths.
CONSONANT_HEIGHT = 5.4


# --------------------------------------------------------------------------- #
# Hearing and harmony
# --------------------------------------------------------------------------- #
def _erb_rate(hz):
    """Hz to ERB-rate: the number of auditory filters below *hz* (Glasberg & Moore)."""
    return 21.4 * np.log10(1 + 0.00437 * np.asarray(hz, dtype=float))


def _erb_spaced(n, low, high):
    """*n* frequencies from *low* to *high* (Hz), evenly spaced to the ear."""
    rate = np.linspace(_erb_rate(low), _erb_rate(high), n)
    return (10 ** (rate / 21.4) - 1) / 0.00437


def _equal_loudness_gain(hz):
    """Gain lifting *hz* halfway to equal loudness (half an inverse A-weighting).

    Half, because A-weighting describes quiet sounds; at listening level the
    ear's bass loss is roughly half as steep. Below 100 Hz the boost stops
    growing: there it would only feed speakers that cannot play it, and the
    overtones of a low voice already carry its pitch.
    """
    f2 = np.maximum(np.asarray(hz, dtype=float), 100) ** 2
    a_weight = (12194 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2)
        * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194 ** 2))
    return (a_weight * 10 ** (2 / 20)) ** -0.5


def _typical(rows):
    """Typical size of each row, robust enough that a spike cannot set the scale."""
    return np.array([np.quantile(np.abs(row), 0.75) for row in rows])


def _equalised(typical, kept):
    """Level of each band after equalisation, relative to the strongest band.

    Flattens the spectrum, leaving a fraction *kept* of its slope, but boosts no
    band by more than ``MAX_BOOST_DB``: a nearly empty band holds little more
    than leakage and sensor noise, which should stay in the background.
    """
    top = typical.max()
    if top == 0:
        return np.zeros_like(typical)
    ratio = typical / top
    return np.minimum(ratio ** kept, ratio * 10 ** (MAX_BOOST_DB / 20))


def _scale_notes(root, ratios, top):
    """Every note of a scale from *root* (Hz) up to *top* (Hz), ascending."""
    octaves = int(np.ceil(np.log2(top / root))) + 1
    notes = root * 2.0 ** np.arange(octaves)[:, None] * np.asarray(ratios)[None, :]
    notes = np.sort(notes.ravel())
    return notes[notes <= top * 1.001]


def _pad_notes(root, ratios):
    """Notes that can be sustained together: chord tones, and only an open bass.

    The least consonant degree of the scale is left to the melody, since held
    next to its neighbours it would be the roughest pair in the chord. Low down,
    where the ear's filters are wide, even a third is muddy.
    """
    chord = sorted(ratios, key=lambda ratio: _harmonic_height(1.0, ratio))[:4]
    upper = _scale_notes(root, chord, PAD_TOP)
    bass = _scale_notes(root, [r for r in chord if r in (1, 3 / 2)], BASS_TOP)
    return np.concatenate([bass, upper[upper > BASS_TOP]])


def _snap_to_notes(targets, notes):
    """Nearest distinct note for each ascending target, while the scale has room."""
    rate = _erb_rate(notes)
    picks = [int(np.abs(rate - _erb_rate(t)).argmin()) for t in targets]
    for k in range(1, len(picks)):
        picks[k] = min(max(picks[k], picks[k - 1] + 1), notes.size - 1)
    return notes[picks]


def _harmonic_height(low, high):
    """Tenney height of an interval reduced to one octave: 0 is a unison.

    The simpler the ratio the more consonant the interval, which ranks
    fifth < fourth < sixth < third < second, as tradition does.
    """
    ratio = Fraction(float(2 ** (np.log2(high / low) % 1))).limit_denominator(64)
    return float(np.log2(ratio.numerator * ratio.denominator))


# --------------------------------------------------------------------------- #
# From the caller's data to a clean signal
# --------------------------------------------------------------------------- #
def _parse_level(level):
    text = str(level).strip().upper().lstrip("L")
    if text not in {"0", "1", "2"}:
        raise ValueError(f"Unknown level {level!r}: use 0, 1 or 2.")
    return int(text)


def _check_duration(duration):
    if duration is not None and duration <= 0:
        raise ValueError(f"`duration` must be positive, got {duration}.")


def _check_correlation_window(correlation_window):
    if correlation_window is not None and correlation_window <= 0:
        raise ValueError(f"`correlation_window` must be positive or None, got {correlation_window}.")


def _parse_scale(scale):
    """Frequency ratios from a name in ``SCALES`` or a tuple of ratios."""
    ratios = SCALES.get(scale, scale) if isinstance(scale, str) else scale
    if isinstance(ratios, str):
        raise ValueError(f"Unknown scale {scale!r}: use one of {list(SCALES)} or a tuple of ratios.")
    return ratios


def _parse_instruments(instruments):
    """List of instrument names from one name, several, or ``None``."""
    chosen = [instruments] if isinstance(instruments, str) else list(instruments or [])
    unknown = [name for name in chosen if name not in INSTRUMENTS]
    if unknown:
        raise ValueError(f"Unknown instrument {unknown[0]!r}: use one of {list(INSTRUMENTS)}.")
    return chosen


def _pick(data, name):
    """``data[name]`` from a Dataset, DataFrame or dict, or fail with the options."""
    if name is None:
        raise ValueError("`variable` is required when `data` holds several variables.")
    if name not in data:
        available = list(data.data_vars) if isinstance(data, xr.Dataset) else list(data)
        raise ValueError(f"Variable {name!r} not found. Available: {available}.")
    return data[name]


def _infer_fs(series):
    """Sampling rate from a 1-D time coordinate, or ``None`` if there is none."""
    if not isinstance(series, xr.DataArray) or series.ndim != 1:
        return None
    coord = series.coords.get(series.dims[0])
    if coord is None or coord.size < 2:
        return None
    step = np.median(np.diff(coord.values))
    if np.issubdtype(coord.dtype, np.datetime64):
        step = step / np.timedelta64(1, "s")
    elif not np.issubdtype(coord.dtype, np.number):
        return None
    return 1 / float(step) if step > 0 else None


def _clean(series):
    """Gap-filled 1-D values of *series* and the mask of what was missing."""
    values = np.asarray(series, dtype=float).ravel()
    values = np.where(np.isfinite(values), values, np.nan)
    if not np.isfinite(values).any():
        raise ValueError("No finite values to sonify.")
    if values.size < 4:
        raise ValueError(f"Need at least 4 samples to sonify, got {values.size}.")
    ready = prepare_signal(xr.DataArray(values, dims="t"))
    return ready["signal"].values, ready["signan"].values.astype(bool)


def _decompose(values, fs, wavelet):
    """Wavelet bands of *values*, finest first, and their frequencies (Hz)."""
    depth = pywt.dwt_max_level(values.size, wavelet)
    if depth < 2:  # too short for this filter; Haar needs only 2 samples
        wavelet = "haar"
        depth = pywt.dwt_max_level(values.size, wavelet)
    result = universal_wt(xr.DataArray(values, dims="t"), method="dwt", fs=fs,
                          wavelet=wavelet, level=depth, coi=False)
    return result["wave"].values, result["natural_frequency"].values


def _band_names(freqs):
    """Name of each band, coarsest first, as stems and MIDI tracks show them."""
    return [f"{f:g} Hz" for f in freqs[::-1]]


def _load_product(data, variable, fs, wavelet):
    """A product as its cross-scalogram, as in ``data_compute_product``.

    That is the wavelet-domain flux density: it sums over scales to the flux.
    """
    names = formula_to_vars(variable).xy
    series = [_pick(data, name) for name in names]
    fs = fs or _infer_fs(series[0]) or 1.0
    cleaned = [_clean(s) for s in series]
    bands = None
    parts = []
    for name, (values, _) in zip(names, cleaned):
        decomposed, freqs = _decompose(values, fs, wavelet)
        parts.append(dict(name=name, values=values,
                          power=np.einsum("ij,ij->j", decomposed, decomposed)))
        bands = decomposed if bands is None else bands * decomposed
    source = dict(values=bands.sum(axis=0), bands=bands, freqs=freqs,
                  gaps=np.any([g for _, g in cleaned], axis=0), fs=fs, name=variable,
                  signed=True)
    if len(names) == 2:
        source["parts"] = parts
    return source


def _load_wavelet_domain(series, fs, name):
    """A variable that is already decomposed (e.g. from ``data_run``)."""
    freqs = series["natural_frequency"].values
    if np.any(np.diff(freqs) >= 0):
        raise ValueError(
            "Wavelet-domain variables must come from the discrete transform "
            "(`method='dwt'`), whose bands run from fine to coarse.")
    values = series.transpose("natural_frequency", ...).values
    bands = values.reshape(freqs.size, -1).astype(float)
    gaps = np.isnan(bands).all(axis=0)
    if gaps.all():
        raise ValueError("No finite values to sonify.")
    bands = np.nan_to_num(bands)
    return dict(values=bands.sum(axis=0), bands=bands, freqs=freqs, gaps=gaps,
                fs=fs or 2 * float(freqs.max()), name=name, signed=True)


def _load(data, variable, fs, wavelet):
    """Resolve what the caller passed.

    Returns:
        dict with ``values`` (1-D), ``gaps`` (bool, same length), ``fs``, ``name``
        and, when the input already implies them, ``bands`` (finest first) and
        ``freqs``. A product of two variables adds ``parts``.
    """
    if variable is not None and "*" in variable:
        return _load_product(data, variable, fs, wavelet)
    is_table = isinstance(data, (xr.Dataset, pd.DataFrame, dict))
    series = _pick(data, variable) if is_table else data
    if isinstance(series, pd.Series):
        series = xr.DataArray(series)
    name = variable or getattr(series, "name", None) or "signal"
    if isinstance(series, xr.DataArray) and "natural_frequency" in series.dims:
        return _load_wavelet_domain(series, fs, name)
    values, gaps = _clean(series)
    return dict(values=values, gaps=gaps, fs=fs or _infer_fs(series) or 1.0, name=name,
                signed=False)


# --------------------------------------------------------------------------- #
# Shared by L1 and L2
# --------------------------------------------------------------------------- #
def _to_control(series, n_control, window):
    """Moving-average *series* over *window* samples, sampled at *n_control* points."""
    smooth = uniform_filter1d(series, max(int(round(window)), 1), mode="nearest")
    return np.interp(np.linspace(0, series.size - 1, n_control),
                     np.arange(series.size), smooth)


def _rms_curve(power, n_control, window):
    """Root of a smoothed power (running sums can dip just below zero in a gap)."""
    return np.sqrt(np.maximum(_to_control(power, n_control, window), 0))


def _presence(gaps, n_control):
    """Curve that is 1 where there are data and 0 across gaps, without clicks."""
    present = _to_control(1.0 - gaps, n_control, gaps.size / n_control)
    return uniform_filter1d(np.clip(4 * present - 2.5, 0, 1), 4, mode="nearest")


def _pan_gains(pan):
    """Constant-power left and right gains for *pan* in -1..1."""
    angle = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return np.cos(angle), np.sin(angle)


# --------------------------------------------------------------------------- #
# L0: audio file
# --------------------------------------------------------------------------- #
def _audify(source, duration, sample_rate, parts=None):
    """The waveform itself: mean removed, peak-normalised, edges declicked.

    Returns:
        Stereo audio, the audible fraction, and the same treatment of each of
        *parts* (series that sum to the signal), scaled to sum to that audio.
    """
    values = source["values"] - source["values"].mean()
    spectrum = np.abs(np.fft.rfft(values)) ** 2
    audio_hz = np.arange(spectrum.size) / duration  # k cycles per record
    total = spectrum[1:].sum()
    audible = spectrum[(audio_hz >= AUDIBLE_RANGE[0]) & (audio_hz <= AUDIBLE_RANGE[1])].sum()
    fraction = float(audible / total) if total > 0 else 0.0
    if fraction < 0.5:
        logger.warning(
            "Only %.0f%% of the variance is audible at this speed; try level=1 "
            "or 2, or a shorter duration.", 100 * fraction)
    if duration < 0.25:
        logger.warning("This record lasts only %.0f ms as audio; pass `duration` "
                       "to slow it down, or use level=1 or 2.", 1000 * duration)

    n_out = int(round(duration * sample_rate))
    taper = spsignal.windows.tukey(values.size, min(1.0, 0.04 / duration))
    present = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, values.size),
                        1.0 - source["gaps"])
    present = uniform_filter1d(present, int(0.005 * sample_rate) or 1, mode="nearest")

    def as_audio(series):
        wave = series * taper
        if n_out != series.size:
            wave = spsignal.resample(wave, n_out)
        return wave * present

    wave = as_audio(values)
    peak = np.abs(wave).max()
    scale = PEAK_CEILING / peak if peak > 0 else 1.0
    stereo = [np.column_stack([as_audio(p) * scale] * 2) for p in parts] if parts is not None else None
    return np.column_stack([wave * scale] * 2), fraction, stereo


# --------------------------------------------------------------------------- #
# L1: audible
# --------------------------------------------------------------------------- #
def _expand_band(band, cycles_per_sample, target, neighbourhood, n_out, sample_rate, rng):
    """Move one wavelet band up to *target* Hz without moving it in time.

    Short grains of the band are read fast enough to sound at *target*, while
    the place they are read from advances at the speed of the record, so pitch
    no longer depends on speed. A grain lasts a few cycles of its band, as a
    wavelet does. Grains come from around the current place, at least a period
    apart so that they do not repeat in phase and buzz, and within
    *neighbourhood* samples so that loudness cannot change faster than that.

    Returns:
        Stereo array of shape ``(n_out, 2)``, grains spread across the field.
    """
    step = target / sample_rate / cycles_per_sample  # data samples per audio sample
    hop = max(int(GRAIN_CYCLES * sample_rate / target / GRAIN_OVERLAP), 1)
    length = hop * GRAIN_OVERLAP
    n_grains = -(-n_out // hop) + GRAIN_OVERLAP
    places = ((np.arange(n_grains) - GRAIN_OVERLAP) * hop + length / 2) / n_out * band.size
    reach = max(1 / cycles_per_sample, neighbourhood)
    places = places + rng.uniform(-reach, reach, n_grains)
    left, right = _pan_gains(rng.uniform(-0.7, 0.7, n_grains))

    window = spsignal.windows.hann(length, sym=False)
    offsets = (np.arange(length) - length / 2) * step
    spline = spline_filter1d(band, order=3, mode="mirror")
    rows = np.zeros((2, n_grains + GRAIN_OVERLAP, hop))
    chunk = max(GRAIN_CHUNK // length, 1)
    for first in range(0, n_grains, chunk):
        last = min(first + chunk, n_grains)
        reads = (places[first:last, None] + offsets).ravel()[None]
        grains = map_coordinates(spline, reads, order=3, mode="mirror", prefilter=False)
        grains = grains.reshape(last - first, length) * window
        for q in range(GRAIN_OVERLAP):  # overlap-add, one quarter of a grain at a time
            part = grains[:, q * hop:(q + 1) * hop]
            rows[0, first + q:last + q] += part * left[first:last, None]
            rows[1, first + q:last + q] += part * right[first:last, None]
    skip = GRAIN_OVERLAP * hop
    return rows.reshape(2, -1)[:, skip:skip + n_out].T


def _render_audible(bands, freqs, source, duration, sample_rate, rng):
    """L1: expand the bands over the hearing range, equalise them, weight them.

    Returns:
        The part of each band, coarsest first, and the frequency it was given.
    """
    bands, freqs = bands[::-1], freqs[::-1]  # coarsest first, like the pitches
    targets = _erb_spaced(bands.shape[0], *HEARING_BAND)
    n_out = int(round(duration * sample_rate))
    typical = _typical(bands)
    levels = _equalised(typical, SLOPE_KEPT)
    neighbourhood = bands.shape[1] / duration / MAX_MODULATION_HZ / 2

    n_control = max(int(duration * CONTROL_RATE), 2)
    gate = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, n_control),
                     _presence(source["gaps"], n_control))[:, None]
    parts = []
    for band, freq, target, size, level in zip(bands, freqs, targets, typical, levels):
        if size == 0:
            parts.append(np.zeros((n_out, 2)))
            continue
        softened = np.tanh(band / (5 * size))  # spikes saturate, the rest is untouched
        # a DWT band spans freq/2..freq, so its centre is 0.75 * freq
        expanded = _expand_band(softened, 0.75 * freq / source["fs"], target, neighbourhood,
                                n_out, sample_rate, rng)
        part = expanded * (level * _equal_loudness_gain(target) / expanded.std())
        parts.append((part - part.mean(axis=0)) * gate)
    return parts, targets


# --------------------------------------------------------------------------- #
# L2: music
# --------------------------------------------------------------------------- #
def _unit(curve):
    """Scale a non-negative curve so that its 95th percentile is 1, clipped."""
    top = np.quantile(curve, 0.95)
    return np.clip(curve / top, 0, 1) if top > 0 else curve


def _value_curve(values, n_control, window):
    """Smoothed value on a robust 0..1 scale (spikes cannot flatten the rest)."""
    curve = _to_control(values, n_control, window)
    low, high = np.quantile(curve, [0.02, 0.98])
    if high <= low:
        return np.full(n_control, 0.5)
    return np.clip((curve - low) / (high - low), 0, 1)


def _perceptual_amplitude(level, kept, ceiling=4.0):
    """Band levels to amplitudes that sound proportional to the data.

    Red spectra would leave only the largest eddies audible, so each band is
    judged against its own typical level and then equalised, keeping a fraction
    *kept* of the real slope so that variables still differ in colour. Since
    loudness grows as pressure ** 0.6, the inverse exponent makes perceived
    loudness follow the data linearly. Spikes saturate softly at *ceiling* times
    the typical level.
    """
    typical = _typical(level)
    relative = np.divide(level, typical[:, None], out=np.zeros_like(level),
                         where=typical[:, None] > 0)
    relative = ceiling * np.tanh(relative / ceiling)
    return relative ** (1 / LOUDNESS_EXPONENT) * _equalised(typical, kept)[:, None]


def _score(source, duration, correlation_window=None):
    """Reduce the signal to the curves that drive the music, in audio time.

    A band of one variable is summarised by its local RMS over at least one of
    its own periods, so a voice follows the energy at that scale rather than
    oscillating with it. A band of a flux is summarised by its local mean over
    several periods, which is the covariance at that scale.

    Args:
        source: Result of ``_load``, with its bands.
        duration: Seconds of sound for the whole record.
        correlation_window: Seconds of data over which a duet's correlation is
            judged. By default ``CORRELATION_SECONDS`` of sound, however much
            data that is.

    Returns:
        dict of curves on a common grid: ``amp`` and ``pan`` per band (coarsest
        first), ``value``, ``activity`` and ``gate``. A duet adds ``parts``
        (``value`` and ``activity`` per variable), ``correlation`` (-1..1),
        ``direction`` (0 downward, 1 upward) and ``harmony`` (0 no more than
        chance, 1 strong).
    """
    bands = source["bands"][::-1]
    freqs = source["freqs"][::-1]
    parts = source.get("parts")
    n_samples = bands.shape[1]
    n_control = max(int(duration * CONTROL_RATE), 2)
    hop = n_samples / n_control

    level = np.empty((bands.shape[0], n_control))
    pan = np.zeros_like(level)
    for j, (band, freq) in enumerate(zip(bands, freqs)):
        # a DWT band spans freq/2..freq, so its typical period is 1.5/freq
        window = max((4 if parts else 1) * 1.5 * source["fs"] / freq, hop)
        mean = _to_control(band, n_control, window)
        if source["signed"]:
            pan[j] = mean / np.maximum(_to_control(np.abs(band), n_control, window), 1e-30)
        level[j] = np.abs(mean) if parts else _rms_curve(band ** 2, n_control, window)
    settle = max(int(CONTROL_RATE / PAD_MOD_HZ), 1)
    level = np.maximum(uniform_filter1d(level, settle, axis=1, mode="nearest"), 0)
    pan = uniform_filter1d(pan, settle, axis=1, mode="nearest")

    gate = _presence(source["gaps"], n_control)
    amp = _perceptual_amplitude(level, FLUX_SLOPE_KEPT if parts else SLOPE_KEPT) * gate
    note_window = max(hop, n_samples / (duration * NOTE_RATE))
    score = dict(amp=amp, pan=pan, gate=gate, activity=_unit(amp.mean(axis=0)),
                 value=_value_curve(source["values"], n_control, note_window))
    if parts:
        window = (CORRELATION_SECONDS * n_samples / duration if correlation_window is None
                  else correlation_window * source["fs"])
        window = max(window, hop)
        covariance = _to_control(source["values"], n_control, window)
        spread = (_rms_curve(parts[0]["power"], n_control, window)
                  * _rms_curve(parts[1]["power"], n_control, window))
        score["correlation"] = np.clip(covariance / np.maximum(spread, 1e-30), -1, 1)
        score["harmony"] = np.clip((np.abs(score["correlation"]) - HARMONY_ONSET)
                                   / (HARMONY_FULL - HARMONY_ONSET), 0, 1)
        typical = np.sqrt((covariance ** 2).mean())
        score["direction"] = (0.5 + 0.5 * np.tanh(2 * covariance / typical) if typical > 0
                              else np.full(n_control, 0.5))
        score["parts"] = [
            dict(value=_value_curve(p["values"], n_control, note_window),
                 activity=_unit(_rms_curve(p["power"], n_control, note_window)))
            for p in parts]
    return score


def _voice(freq, amp, t, rng):
    """A soft sustained tone whose overtones open up as it gets louder."""
    harmonics = int(np.clip(2000 // freq, 1, 5))
    fundamental = np.zeros_like(t)
    overtones = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        # A quiet detuned twin gives slow shimmer without cancelling the note.
        partial = (np.sin(2 * np.pi * h * freq * t + rng.uniform(0, 2 * np.pi))
                   + 0.35 * np.sin(2 * np.pi * h * freq * 1.0007 * t + rng.uniform(0, 2 * np.pi)))
        if h == 1:
            fundamental += partial
        else:
            overtones += partial / h ** 2
    return amp * (fundamental + overtones * np.sqrt(np.minimum(amp, 1.0)))


def _pad_voices(score, ratios, root):
    """Frequency of each band's voice, coarsest lowest; a duet leaves an octave to the bass."""
    root = 2 * root if "parts" in score else root
    return _snap_to_notes(_erb_spaced(score["amp"].shape[0], root, PAD_TOP), _pad_notes(root, ratios))


def _pad(score, n_out, sample_rate, rng, ratios, root):
    """Band *j* of the signal drives the loudness of voice *j*, coarsest lowest."""
    amp, pan = score["amp"], score["pan"]
    n_bands = amp.shape[0]
    voices = _pad_voices(score, ratios, root)
    t = np.arange(n_out) / sample_rate
    grid = np.linspace(0, t[-1], amp.shape[1])
    parts = []
    for j, freq in enumerate(voices):
        # Bass stays centred; upper voices alternate outwards.
        spread = 0.7 * (j / max(n_bands - 1, 1)) * (-1) ** j
        tone = _voice(freq, np.interp(t, grid, amp[j]), t, rng) * _equal_loudness_gain(freq)
        left, right = _pan_gains(spread + 0.5 * np.interp(t, grid, pan[j]))
        parts.append(np.column_stack([left * tone, right * tone]))
    return parts, voices


def _bass_notes(root, ratios):
    """Frequencies of the bass: for upward flux, then for downward."""
    return root, root * ratios[-1] / 2


def _bass(score, n_out, sample_rate, rng, ratios, root):
    """Lowest voice, which names the direction of the flux.

    It sits on the root for upward flux and on the top degree of the scale, an
    octave down, for downward flux. Both belong to the scale, so the change is
    one of colour, not of key: in the default scale, major against relative minor.
    """
    t = np.arange(n_out) / sample_rate
    grid = np.linspace(0, t[-1], score["gate"].size)
    upward = np.interp(t, grid, score["direction"])
    level = np.interp(t, grid, score["activity"] * score["gate"])
    up, down = _bass_notes(root, ratios)
    tone = (_voice(up, level * upward, t, rng) * _equal_loudness_gain(up)
            + _voice(down, level * (1 - upward), t, rng) * _equal_loudness_gain(down))
    return np.column_stack([tone, tone]) / np.sqrt(2)


def _slot_seconds(n_samples, duration):
    """Time between melody slots: never more than one slot per sample."""
    return 1 / min(NOTE_RATE, n_samples / duration)


def _note_grid(n_samples, n_control, duration):
    """Times of the melody slots (s) and their place on the control grid.

    Never more than one slot per sample, so a short series plays its own values.
    """
    slots = np.arange(0, duration, _slot_seconds(n_samples, duration))
    return slots, np.minimum((slots / duration * n_control).astype(int), n_control - 1)


def _line(curve, n_notes):
    """Note indices for a 0..1 curve, moving by step like a voice would."""
    wanted = np.round(curve * (n_notes - 1)).astype(int)
    path = np.empty_like(wanted)
    path[0] = wanted[0]
    for s in range(1, wanted.size):
        path[s] = path[s - 1] + np.clip(wanted[s] - path[s - 1], -2, 2)
    return path


def _harmonise(lead, wanted, strength, lead_notes, notes):
    """The second line: its own notes, pulled towards consonance by *strength*.

    At each slot it may step at most two notes. Of those it takes the closest to
    *wanted* when *strength* is 0, and the closest that is consonant with *lead*
    when it is 1, moving further for a stronger pull. It looks for the closest
    rather than the most consonant so that the line keeps its own contour:
    otherwise it would shadow the lead in octaves, and two anti-correlated
    variables would not sound like contrary motion.

    Args:
        lead: Note index of the first line per slot, in *lead_notes*.
        wanted: Fractional note index of this line per slot, in *notes*.
        strength: 0..1 per slot.
        lead_notes, notes: Frequencies of each line's notes (Hz).

    Returns:
        Note index of this line per slot.
    """
    clash = np.array([[_harmonic_height(a, b) > CONSONANT_HEIGHT for b in notes]
                      for a in lead_notes])
    path = np.empty(wanted.size, dtype=int)
    current = int(round(wanted[0]))
    for s in range(wanted.size):
        options = np.arange(max(current - 2, 0), min(current + 2, notes.size - 1) + 1)
        cost = np.abs(options - wanted[s]) / 3 + strength[s] * clash[lead[s], options]
        current = int(options[np.argmin(cost)])
        path[s] = current
    return path


def _duet(score, grid, lead_notes, notes):
    """Note paths of two instruments whose relation is the local correlation.

    The second line is the Cholesky mix ``r * a + sqrt(1 - r ** 2) * b``. Flux is
    carried by scales that may be far faster than the melody; this gives the two
    lines, at the speed of the melody, the correlation the variables have at the
    speed of the turbulence.
    """
    first, second = (2 * p["value"][grid] - 1 for p in score["parts"])
    correlation = score["correlation"][grid]
    mixed = correlation * first + np.sqrt(1 - correlation ** 2) * second
    lead = _line((first + 1) / 2, lead_notes.size)
    wanted = (np.clip(mixed, -1, 1) + 1) / 2 * (notes.size - 1)
    return lead, _harmonise(lead, wanted, score["harmony"][grid], lead_notes, notes)


def _note(freq, seconds, timbre, sample_rate):
    """One note of an instrument, with no click at either end."""
    t = np.arange(int((seconds + timbre["release"]) * sample_rate)) / sample_rate
    wobble = (timbre["vibrato"] * freq / 5 * np.minimum(t / 0.4, 1)
              * np.sin(2 * np.pi * 5 * t))  # 5 Hz, fading in
    top = 0.45 * sample_rate  # anything above Nyquist would fold back out of tune
    tone = sum(weight * np.exp(-t / decay) * np.sin(harmonic * (2 * np.pi * freq * t + wobble))
               for harmonic, weight, decay in timbre["partials"]
               if harmonic == 1 or harmonic * freq < top)
    rise = min(int(timbre["attack"] * sample_rate), t.size)
    fall = min(int(timbre["release"] * sample_rate), t.size)
    tone[:rise] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(rise) / max(rise, 1))
    tone[t.size - fall:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(fall) / max(fall, 1))
    return tone


def _events(path, slots, grid, gate, activity):
    """Note-ons: a note sounds when the pitch changes and there are data."""
    events = []
    previous = None
    for index, when, at in zip(path, slots, grid):
        if index != previous and gate[at] > 0.5:
            events.append((float(when), int(index), 0.45 + 0.55 * float(activity[at])))
        previous = index
    return events


def _cast(source, chosen):
    """Who plays at L2: the name of each line and its instrument."""
    players = [p["name"] for p in source.get("parts", [])] or [source["name"]]
    chosen = chosen or (["flute", "bell"] if len(players) == 2 else ["bell"])
    if len(chosen) != len(players):
        raise ValueError(f"{players} takes {len(players)} instrument(s), got {chosen}.")
    return players, chosen


def _melody(source, score, duration, ratios, root, timbres):
    """What each instrument plays, whether it ends up as audio or as MIDI.

    Returns:
        ``(events, notes, timbre, pan)`` per line: the note-ons of ``_events``,
        the frequency of each note they point to, and where the line sits.
    """
    slots, grid = _note_grid(source["values"].size, score["gate"].size, duration)
    if "parts" in score:
        low_notes = _scale_notes(2 * root, ratios, 8 * root)
        high_notes = _scale_notes(4 * root, ratios, 16 * root)
        lead, follower = _duet(score, grid, low_notes, high_notes)
        lines = [(lead, low_notes, timbres[0], -0.35, score["parts"][0]["activity"]),
                 (follower, high_notes, timbres[1], 0.35, score["parts"][1]["activity"])]
    else:
        notes = _scale_notes(4 * root, ratios, 16 * root)
        lines = [(_line(score["value"][grid], notes.size), notes, timbres[0], 0.0,
                  score["activity"])]
    return [(_events(path, slots, grid, score["gate"], activity), notes, timbre, pan)
            for path, notes, timbre, pan, activity in lines]


def _hold(events, k, timbre, duration):
    """Seconds note *k* lasts: its own ring, or until the next one within ``MAX_HOLD``."""
    following = events[k + 1][0] if k + 1 < len(events) else duration
    return timbre["ring"] or min(following - events[k][0], MAX_HOLD)


def _play(events, notes, timbre, pan, n_total, duration, sample_rate):
    """Render a list of note-ons as a stereo stem."""
    stem = np.zeros((n_total, 2))
    left, right = _pan_gains(pan)
    for k, (when, index, velocity) in enumerate(events):
        tone = velocity * _note(notes[index] * 2.0 ** timbre["octave"],
                                _hold(events, k, timbre, duration), timbre, sample_rate)
        start = int(when * sample_rate)
        stop = min(start + tone.size, n_total)
        stem[start:stop, 0] += left * tone[:stop - start]
        stem[start:stop, 1] += right * tone[:stop - start]
    return stem


def _room(sample_rate, rng, decay=2.4):
    """Impulse of a decorrelated stereo room: a bright early tail and a darker late one."""
    t = np.arange(int(1.3 * decay * sample_rate)) / sample_rate
    noise = rng.standard_normal((t.size, 2))
    dark = spsignal.sosfilt(
        spsignal.butter(2, 2500, "low", fs=sample_rate, output="sos"), noise, axis=0)
    room = (0.5 * noise * np.exp(-6.9 * t / (0.25 * decay))[:, None]
            + dark * np.exp(-6.9 * t / decay)[:, None])
    room /= np.sqrt((room ** 2).sum(axis=0))
    return np.vstack([np.zeros((int(0.02 * sample_rate), 2)), room])


def _in_room(dry, room, wet=0.3):
    """*dry* heard in *room*. Linear, so parts can go in one by one."""
    return dry + wet * spsignal.fftconvolve(dry, room, axes=0)[:dry.shape[0]]


def _render_music(source, score, duration, sample_rate, rng, ratios, root, timbres):
    """L2: voices for the scales, one instrument per variable above them.

    Returns:
        Dry stereo parts and their names, voices first (coarsest first), and the
        frequency of each voice.
    """
    n_out = int(round(duration * sample_rate))
    n_total = n_out + int(RING_OUT * sample_rate)

    pad, voices = _pad(score, n_out, sample_rate, rng, ratios, root)
    harmony = 1.0
    extra = []
    if "parts" in score:
        extra = [_bass(score, n_out, sample_rate, rng, ratios, root)]
        # Each band is judged against itself, so the voices alone could not tell
        # a strong flux from none. They only open up as far as the covariance.
        harmony = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, score["gate"].size),
                            score["harmony"])[:, None]

    # Balance by level, so the lines sit above the pad whatever the data do.
    level = np.sum(pad + extra, axis=0).std()
    scale = 1 / level if level > 0 else 0.0
    parts = []
    for part in pad + extra:
        padded = np.zeros((n_total, 2))
        padded[:n_out] = part * scale * harmony
        parts.append(padded)
    names = [f"voice {j}" for j in range(len(pad))] + ["bass"] * len(extra)
    lines = _melody(source, score, duration, ratios, root, timbres)
    for k, (events, notes, timbre, pan) in enumerate(lines):
        stem = _play(events, notes, timbre, pan, n_total, duration, sample_rate)
        level = stem.std()
        parts.append(stem * (LINE_LEVEL / level) if level > 0 else stem)
        names.append(f"line {k}")
    return parts, names, voices


# --------------------------------------------------------------------------- #
# Mastering
# --------------------------------------------------------------------------- #
def _k_weighting(sample_rate):
    """The two BS.1770 pre-filters, for any sample rate, as second-order sections.

    A high shelf for the head, then a high-pass for low-frequency insensitivity.
    These analogue parameters reproduce the 48 kHz coefficients of the standard.
    """
    k = np.tan(np.pi * 1681.9744509555319 / sample_rate)
    q = 0.7071752369554193
    high = 10 ** (3.99984385397 / 20)
    mid = high ** 0.499666774155
    shelf = [high + mid * k / q + k * k, 2 * (k * k - high), high - mid * k / q + k * k,
             1 + k / q + k * k, 2 * (k * k - 1), 1 - k / q + k * k]
    k = np.tan(np.pi * 38.13547087613982 / sample_rate)
    q = 0.5003270373253953
    scale = 1 + k / q + k * k
    high_pass = [scale, -2 * scale, scale, scale, 2 * (k * k - 1), 1 - k / q + k * k]
    sections = np.array([shelf, high_pass])
    return sections / sections[:, 3:4]


def _weighted_power(audio, sample_rate):
    """K-weighted power of every sample, summed over channels."""
    audio = np.asarray(audio, dtype=float)
    if audio.ndim == 1:
        audio = audio[:, None]
    return (spsignal.sosfilt(_k_weighting(sample_rate), audio, axis=0) ** 2).sum(axis=1)


def integrated_loudness(audio, sample_rate=SAMPLE_RATE):
    """Integrated programme loudness (ITU-R BS.1770-4), in LUFS.

    Args:
        audio: Array of shape ``(samples,)`` or ``(samples, channels)``.
        sample_rate: Sample rate in Hz.

    Returns:
        Loudness in LUFS; ``-inf`` for silence.
    """
    power = _weighted_power(audio, sample_rate)

    def lufs(mean_square):
        with np.errstate(divide="ignore"):
            return -0.691 + 10 * np.log10(mean_square)

    block = int(0.4 * sample_rate)
    if power.size < block:
        return float(lufs(power.mean()))
    running = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, power.size - block + 1, block // 4)
    blocks = (running[starts + block] - running[starts]) / block
    blocks = blocks[lufs(blocks) > -70]  # absolute gate
    if blocks.size == 0:
        return -np.inf
    blocks = blocks[lufs(blocks) > lufs(blocks.mean()) - 10]  # relative gate
    return float(lufs(blocks.mean()))


def _limit_gain(audio, sample_rate):
    """Look-ahead limiter: turns peaks down smoothly instead of clipping them."""
    reach = 2 * int(0.01 * sample_rate) + 1
    held = maximum_filter1d(np.abs(audio).max(axis=1), reach, mode="nearest")
    gain = PEAK_CEILING / np.maximum(held, PEAK_CEILING)
    return uniform_filter1d(gain, reach, mode="nearest")


def _swing_gain(audio, sample_rate, n_record, widest):
    """Gain that brings the slow swings of loudness within *widest* LU, keeping their order.

    A long record can hold a calm night and a turbulent noon, so far apart that
    the night is lost when the noon sounds right. Soft passages stay softer than
    loud ones, only by less, and notes keep their own dynamics because the gain
    moves over seconds.

    Args:
        audio: Stereo array.
        sample_rate: Sample rate in Hz.
        n_record: Samples that carry the record. Whatever follows, such as the
            last notes ringing out, has no say and takes the last gain.
        widest: Largest swing in LU, from the 5th to the 95th percentile.

    Returns:
        Gain per sample, or ``None`` if the record already fits.
    """
    window = int(LOUDNESS_WINDOW * sample_rate)
    if n_record < window:  # too short to have slow swings
        return None
    hop = max(window // 30, 1)
    power = _weighted_power(audio[:n_record], sample_rate)
    blocks = power[:power.size // hop * hop].reshape(-1, hop).mean(axis=1)
    if not blocks.any():
        return None
    # A gap has no loudness to keep, so windows average only the sound beside it.
    sounding = blocks > 1e-6 * blocks.max()
    share = uniform_filter1d(sounding.astype(float), 30, mode="nearest")
    known = share > 0.02
    mean = uniform_filter1d(blocks * sounding, 30, mode="nearest")
    level = 10 * np.log10(np.maximum(mean[known], 1e-30) / share[known])
    low, high = np.quantile(level, [0.05, 0.95])
    if high - low <= widest:
        return None
    gain = (np.clip(level, low, high) - (low + high) / 2) * (widest / (high - low) - 1)
    gain = uniform_filter1d(gain, 30, mode="nearest")
    gain = np.interp(np.arange(audio.shape[0]), (np.flatnonzero(known) + 0.5) * hop, gain)
    return 10 ** (gain / 20)


def _master(audio, sample_rate, target, n_record, widest):
    """Narrow slow swings, fade the edges, reach *target* and keep the peaks legal.

    Returns:
        The mastered audio and the gain per sample that made it. Every step is a
        gain, so parts of the mix can take the same curve and still sum to it.
    """
    gain = np.ones(audio.shape[0])
    if widest is not None:
        swing = _swing_gain(audio, sample_rate, n_record, widest)
        if swing is not None:
            gain = swing
    fade = min(int(1.5 * sample_rate), audio.shape[0] // 4)
    if fade:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
        gain[:fade] *= ramp
        gain[-fade:] *= ramp[::-1]
    for _ in range(3):  # limiting lowers the loudness a little, so converge
        measured = integrated_loudness(audio * gain[:, None], sample_rate)
        if not np.isfinite(measured):
            break
        gain = gain * 10 ** ((target - measured) / 20)
        gain = gain * _limit_gain(audio * gain[:, None], sample_rate)
    return audio * gain[:, None], gain


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def sonify(data, variable=None, level=2, duration=None, fs=None,
           scale="pentatonic", root=65.41, instruments=None, loudness=-16.0,
           loudness_range=6.0, sample_rate=SAMPLE_RATE, wavelet="db6", seed=0, dst=None,
           stems=False, correlation_window=None):
    """Turn a variable into sound.

    Args:
        data: The signal: a 1-D array, ``pandas.Series`` or
            ``xarray.DataArray``; or a Dataset, DataFrame or dict holding
            *variable*. Multi-dimensional variables are read in time order, so
            ``(TIMESTAMP, ns)`` blocks play one after another. Variables with a
            ``natural_frequency`` dimension (``data_run`` output such as
            ``wco2`` or ``wco2-wh2o+``) are used as they are.
        variable: Name of the variable, or a product such as ``"w*co2"``. At
            level 2, a product of two variables is played as a duet.
        level: 0 audio file, 1 audible, 2 music (see the module docstring).
            Also accepts ``"L0"``, ``"L1"``, ``"L2"``.
        duration: Seconds of sound for the whole record. By default 30, or at
            level 0 one audio sample per data sample.
        fs: Sampling rate of the data (Hz). Only labels the output: the sound
            does not depend on it. Inferred from a time coordinate if possible.
        scale: L2 only. A name in ``SCALES`` or a tuple of frequency ratios.
        root: L2 only. Lowest note in Hz (default C2).
        instruments: L2 only. A name in ``INSTRUMENTS``, or one per variable of a
            duet, in the order of the product. By default ``"bell"``, and
            ``("flute", "bell")`` for a duet.
        loudness: L1 and L2 target loudness in LUFS.
        loudness_range: L1 and L2 widest slow swing of loudness, in LU. Quiet
            passages stay quieter, but no further below the loud ones than this:
            6 LU sounds about two thirds as loud. ``None`` keeps the full swing.
        sample_rate: Audio sample rate in Hz.
        wavelet: Mother wavelet for the decomposition.
        seed: Seed of the grains, phases and room. The same data and seed always
            give the same sound.
        dst: Optional path of a ``.wav`` file to write (the full mix).
        stems: Return one track per part instead of the mix: every wavelet
            scale, plus the trend at L0 and the melody lines at L2. They share
            one mastering, so they sum to the mix.
        correlation_window: L2 duet only. Seconds of data over which the local
            correlation is judged, e.g. ``1800`` for the eddy-covariance half
            hour. By default 3 s of sound, however much data that is.

    Returns:
        xarray.DataArray with dims ``(time, channel)`` in the range -1..1, or
        ``(stem, time, channel)`` with *stems*.
        ``attrs`` records the mapping: ``level``, ``sample_rate``, ``variable``,
        ``fs``, ``speedup`` and, for L1 and L2, ``band_frequency_hz`` and the
        ``voice_frequency_hz`` each band was given (coarsest first). L0 adds
        ``audible_fraction``, the share of the variance within hearing range. L2
        adds ``instruments``.

    Raises:
        ValueError: If the level, scale, instrument or variable is unknown, the
            number of instruments is wrong, or there are not enough finite values.
    """
    level = _parse_level(level)
    _check_duration(duration)
    if loudness_range is not None and loudness_range <= 0:
        raise ValueError(f"`loudness_range` must be positive or None, got {loudness_range}.")
    _check_correlation_window(correlation_window)
    ratios = _parse_scale(scale)
    chosen = _parse_instruments(instruments)

    source = _load(data, variable, fs, wavelet)
    values = source["values"]
    if duration is None:
        duration = values.size / sample_rate if level == 0 else DEFAULT_DURATION
    rng = np.random.default_rng(seed)
    attrs = dict(level=level, sample_rate=sample_rate, variable=source["name"],
                 fs=source["fs"], speedup=values.size / source["fs"] / duration)

    parts = names = None
    if np.ptp(values) == 0:
        logger.warning("%s is constant: there is nothing to hear.", source["name"])
        audio = np.zeros((int(round(duration * sample_rate)), 2))
    elif level == 0:
        if stems:
            bands, freqs = _decompose(values, source["fs"], wavelet)
            pieces = [values - bands.sum(axis=0) - values.mean()] + list(bands[::-1])
            names = ["trend"] + _band_names(freqs)
        else:
            pieces = None
        audio, attrs["audible_fraction"], parts = _audify(source, duration, sample_rate, pieces)
    else:
        if level == 1:
            bands, freqs = _decompose(values, source["fs"], wavelet)
            parts, voices = _render_audible(bands, freqs, source, duration, sample_rate, rng)
            names = _band_names(freqs)
            audio = np.sum(parts, axis=0)
        else:
            if "bands" not in source:
                source["bands"], source["freqs"] = _decompose(values, source["fs"], wavelet)
            freqs = source["freqs"]
            players, chosen = _cast(source, chosen)
            parts, names, voices = _render_music(source, _score(source, duration, correlation_window),
                                                 duration, sample_rate, rng, ratios, root,
                                                 [INSTRUMENTS[n] for n in chosen])
            room = _room(sample_rate, rng)
            audio = _in_room(np.sum(parts, axis=0), room)
            if stems:
                parts = [_in_room(part, room) for part in parts]
                bands_named = _band_names(freqs)
                names = [bands_named[int(n.split()[1])] if n.startswith("voice") else
                         f"line: {players[int(n.split()[1])]}" if n.startswith("line") else n
                         for n in names]
            attrs["instruments"] = [f"{player}: {name}" for player, name in zip(players, chosen)]
        audio, gain = _master(audio, sample_rate, loudness, int(round(duration * sample_rate)),
                              loudness_range)
        if stems:
            parts = [part * gain[:, None] for part in parts]
        attrs["band_frequency_hz"] = freqs[::-1].tolist()
        attrs["voice_frequency_hz"] = np.asarray(voices).tolist()

    time = np.arange(audio.shape[0]) / sample_rate
    if stems and parts is not None:
        return xr.DataArray(
            np.array(parts, dtype="float32"), dims=("stem", "time", "channel"), name=source["name"],
            coords={"stem": names, "time": time, "channel": ["left", "right"]}, attrs=attrs)
    result = xr.DataArray(
        audio.astype("float32"), dims=("time", "channel"), name=source["name"],
        coords={"time": time, "channel": ["left", "right"]}, attrs=attrs)
    if dst:
        write_wav(result, dst)
    return result


def write_wav(audio, path, sample_rate=None):
    """Write audio to a 16-bit ``.wav`` file.

    Args:
        audio: Result of :func:`sonify`, or an array of samples in -1..1.
        path: Destination path; missing folders are created.
        sample_rate: Sample rate in Hz. Defaults to the one stored in ``audio``.
    """
    rate = sample_rate or getattr(audio, "attrs", {}).get("sample_rate", SAMPLE_RATE)
    samples = np.asarray(audio, dtype=float)
    if samples.any():
        # Triangular dither keeps quiet reverb tails from turning into distortion.
        rng = np.random.default_rng(0)
        samples = samples + (rng.random(samples.shape) - rng.random(samples.shape)) / 32767
    pcm = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
    if os.path.dirname(str(path)):
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    wavfile.write(str(path), int(rate), pcm)
    logger.info("Saved %s (%.1f s).", path, pcm.shape[0] / rate)
