"""Tests for waveletec.extra.sonification.

Whether something sounds good cannot be unit-tested. What can be tested is that
each level does what it claims: L0 is the signal, L1 moves it into the hearing
range without moving it in time, and in L2 covariance becomes harmony.
"""
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import signal as spsignal
from scipy.io import wavfile

import fluxsound
import waveletec
from fluxsound import cli, midi
from fluxsound import sonification as so
from fluxsound.decomposition import decompose

# Numerical trouble here goes quiet instead of crashing: a NaN in one band once
# silenced every sustained voice while the rest of the audio stayed finite.
pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

RATE = 22050


def red_noise(n=2 ** 14, seed=0, slope=-5 / 3):
    """Turbulence-like signal: random phases under a power-law spectrum."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n)
    spectrum = np.zeros(freqs.size, dtype=complex)
    spectrum[1:] = freqs[1:] ** (slope / 2) * np.exp(2j * np.pi * rng.random(freqs.size - 1))
    signal = np.fft.irfft(spectrum, n)
    return signal / signal.std()


def correlated(r, n=2 ** 14, slope=-5 / 3):
    """Two variables with correlation *r* at every scale, as a table."""
    a, b = red_noise(n, 1, slope), red_noise(n, 2, slope)
    return {"a": a, "b": r * a + np.sqrt(1 - r ** 2) * b}


def share(audio, low, high, rate=RATE):
    """Fraction of the power of *audio* between *low* and *high* Hz."""
    freqs, power = spsignal.welch(np.asarray(audio).mean(axis=1), fs=rate, nperseg=4096)
    return power[(freqs >= low) & (freqs < high)].sum() / power.sum()


def rms(audio, start, stop, rate=RATE):
    """RMS of *audio* between two times in seconds."""
    return float(np.sqrt((np.asarray(audio)[int(start * rate):int(stop * rate)] ** 2).mean()))


# In the middle of the band 0.0625-0.125 cycles per sample. A tone on the edge of
# a band would be split between two voices.
BAND_CENTRE = 0.09375


def duet_lines(table, duration=30.0, correlation_window=None):
    """The notes of both instruments, slot by slot, as ``_melody`` builds them."""
    source = so._load(table, "a*b", 1.0, "db6")
    score = so._score(source, duration, correlation_window)
    root, ratios = 65.41, so.SCALES["pentatonic"]
    low = so._scale_notes(2 * root, ratios, 8 * root)
    high = so._scale_notes(4 * root, ratios, 16 * root)
    _, grid = so._note_grid(source["values"].size, score["gate"].size, duration)
    lead, follower = so._duet(score, grid, low, high)
    return score, low[lead], high[follower], lead, follower


def consonant_share(first, second):
    heights = [so._harmonic_height(a, b) for a, b in zip(first, second)]
    return np.mean(np.array(heights) <= so.CONSONANT_HEIGHT)


def parallel_share(lead, follower):
    """Of the slots where both lines move, how many go the same way."""
    a, b = np.sign(np.diff(lead)), np.sign(np.diff(follower))
    both = (a != 0) & (b != 0)
    return np.mean(a[both] == b[both])


# --------------------------------------------------------------------------- #
# L0: the signal as an audio file
# --------------------------------------------------------------------------- #
def test_level0_is_the_signal_itself():
    signal = red_noise() + 400  # a large mean, like CO2 in ppm
    audio = so.sonify(signal, level=0)
    assert audio.sizes["time"] == signal.size  # one audio sample per data sample
    np.testing.assert_array_equal(audio.values[:, 0], audio.values[:, 1])
    inner = slice(1000, -1000)  # away from the declicked edges
    assert np.corrcoef(audio.values[inner, 0], signal[inner])[0, 1] > 0.9999
    assert np.abs(audio.values).max() == pytest.approx(so.PEAK_CEILING, abs=1e-4)


def test_level0_warns_that_slowing_down_drops_the_signal_below_hearing(caplog):
    signal = red_noise(slope=-1)
    fast = so.sonify(signal, level=0)
    assert fast.attrs["audible_fraction"] > 0.5
    with caplog.at_level("WARNING"):
        slow = so.sonify(signal, level=0, duration=120, sample_rate=RATE)
    assert slow.attrs["audible_fraction"] < 0.5 < fast.attrs["audible_fraction"]
    assert "audible" in caplog.text
    assert slow.sizes["time"] == 120 * RATE


# --------------------------------------------------------------------------- #
# L1: altered for hearing
# --------------------------------------------------------------------------- #
def test_level1_reaches_the_hearing_range_that_level0_cannot():
    signal = red_noise()
    raw = so.sonify(signal, level=0, duration=10, sample_rate=RATE)
    heard = so.sonify(signal, level=1, duration=10, sample_rate=RATE)
    assert share(raw, 1000, 10000) < 1e-3
    assert share(heard, 1000, 10000) > 0.05
    assert share(raw, 0, 20) > 3 * share(heard, 0, 20)


def test_level1_keeps_fine_scales_above_large_ones():
    audio = so.sonify(red_noise(), level=1, duration=1, fs=20, sample_rate=RATE)
    bands, targets = audio.attrs["band_frequency_hz"], audio.attrs["voice_frequency_hz"]
    assert bands == sorted(bands) and targets == sorted(targets)
    assert targets[0] == pytest.approx(so.HEARING_BAND[0])
    assert targets[-1] == pytest.approx(so.HEARING_BAND[1])


def test_level1_equalises_without_erasing_the_colour_of_a_variable():
    def centroid(slope):
        audio = so.sonify(red_noise(slope=slope), level=1, duration=4, sample_rate=RATE)
        freqs, power = spsignal.welch(audio.values.mean(axis=1), fs=RATE, nperseg=4096)
        return (freqs * power).sum() / power.sum()

    steep, flat = centroid(-3), centroid(0)
    assert steep > 200  # unequalised, only the lowest band (100 Hz) would be left
    assert steep < 0.8 * flat


def test_level1_keeps_a_pure_tone_pure():
    # Equalisation must not lift wavelet leakage into a band of noise.
    tone = np.sin(2 * np.pi * BAND_CENTRE * np.arange(2 ** 14))
    audio = so.sonify(tone, level=1, duration=4, fs=1.0, sample_rate=RATE)
    bands = np.array(audio.attrs["band_frequency_hz"])
    target = audio.attrs["voice_frequency_hz"][int(np.argmin(np.abs(bands * 0.75 - BAND_CENTRE)))]
    assert share(audio.values, target / 2, target * 2) > 0.8


# --------------------------------------------------------------------------- #
# Both altered levels stay true to the data
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("level", [1, 2])
def test_an_event_is_heard_when_and_where_it_happened(level):
    n, duration = 2 ** 14, 8
    t = np.arange(n)
    signal = 0.02 * red_noise(n)
    during = (t > 0.4 * n) & (t < 0.6 * n)
    signal[during] += np.sin(2 * np.pi * BAND_CENTRE * t[during])

    # Without levelling, which by design lets an event stand out by less.
    audio = so.sonify(signal, level=level, duration=duration, fs=1.0, sample_rate=RATE,
                      loudness_range=None)
    bands = np.array(audio.attrs["band_frequency_hz"])
    voice = audio.attrs["voice_frequency_hz"][int(np.argmin(np.abs(bands * 0.75 - BAND_CENTRE)))]
    sections = spsignal.butter(2, [voice / 1.15, voice * 1.15], "bandpass", fs=RATE, output="sos")
    filtered = spsignal.sosfilt(sections, audio.values, axis=0)
    assert rms(filtered, 3.6, 4.4) > 4 * rms(filtered, 1.5, 2.5)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_gaps_are_silent(level):
    signal = red_noise()
    signal[6000:10000] = np.nan  # from 2.9 s to 4.9 s of 8 s
    audio = so.sonify(signal, level=level, duration=8, sample_rate=RATE)
    assert np.isfinite(audio.values).all()
    if level == 2:  # it keeps ringing into a gap, as a room would, but no note starts there
        assert rms(audio.values, 4.4, 4.8) < 0.3 * rms(audio.values, 1.5, 2.5)
    else:
        assert rms(audio.values, 4.4, 4.8) < 1e-3 * rms(audio.values, 1.5, 2.5)


def test_a_gap_leaves_the_rest_of_the_record_intact():
    signal = red_noise()
    gapped = signal.copy()
    gapped[6000:10000] = np.nan
    sections = spsignal.butter(4, 120, "lowpass", fs=RATE, output="sos")  # the sustained bass
    whole, holed = (spsignal.sosfilt(sections, so.sonify(s, duration=8, sample_rate=RATE).values,
                                     axis=0) for s in (signal, gapped))
    assert rms(holed, 1.5, 2.5) > 0.5 * rms(whole, 1.5, 2.5)


def test_a_gap_in_one_variable_silences_the_duet():
    table = correlated(0.8)
    table["b"][6000:10000] = np.nan
    audio = so.sonify(table, "a*b", duration=8, sample_rate=RATE)
    assert np.isfinite(audio.values).all()
    assert rms(audio.values, 4.4, 4.8) < 0.3 * rms(audio.values, 1.5, 2.5)


@pytest.mark.parametrize("level", [1, 2])
def test_a_spike_does_not_drown_the_record(level):
    signal = red_noise()
    spiked = signal.copy()
    spiked[8000] = 1000.0
    clean = so.sonify(signal, level=level, duration=6, sample_rate=RATE)
    audio = so.sonify(spiked, level=level, duration=6, sample_rate=RATE)
    assert rms(audio.values, 0.5, 2) > 0.5 * rms(clean.values, 0.5, 2)


@pytest.mark.parametrize("level", [1, 2])
def test_the_sound_depends_on_the_data_and_the_seed_only(level):
    signal = red_noise()
    kwargs = dict(level=level, duration=2, sample_rate=RATE)
    first = so.sonify(signal, seed=3, **kwargs).values
    np.testing.assert_array_equal(first, so.sonify(signal, seed=3, **kwargs).values)
    assert not np.allclose(first, so.sonify(signal, seed=4, **kwargs).values)
    assert not np.allclose(first, so.sonify(red_noise(seed=9), seed=3, **kwargs).values)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_fs_labels_the_output_without_changing_the_sound(level):
    signal = red_noise(2 ** 12)
    fast = so.sonify(signal, level=level, duration=2, fs=20, sample_rate=RATE)
    slow = so.sonify(signal, level=level, duration=2, fs=1 / 1800, sample_rate=RATE)
    np.testing.assert_array_equal(fast.values, slow.values)
    assert slow.attrs["speedup"] == pytest.approx(fast.attrs["speedup"] * 20 * 1800)


# --------------------------------------------------------------------------- #
# L2: harmony
# --------------------------------------------------------------------------- #
def test_harmonic_height_ranks_intervals_as_the_ear_does():
    ranked = [1, 3 / 2, 4 / 3, 5 / 3, 5 / 4, 6 / 5, 9 / 8]  # unison, fifth, ... second
    heights = [so._harmonic_height(1.0, ratio) for ratio in ranked]
    assert heights == sorted(heights)
    assert so._harmonic_height(100.0, 200.0) == 0  # an octave counts as a unison
    assert so._harmonic_height(1.0, 6 / 5) <= so.CONSONANT_HEIGHT < so._harmonic_height(1.0, 9 / 8)


@pytest.mark.parametrize("scale", list(so.SCALES))
def test_every_voice_is_a_note_of_the_scale(scale):
    audio = so.sonify(red_noise(), scale=scale, root=55.0, duration=1, sample_rate=RATE)
    notes = so._scale_notes(55.0, so.SCALES[scale], so.PAD_TOP)
    for voice in audio.attrs["voice_frequency_hz"]:
        assert np.abs(notes - voice).min() < 1e-6


def test_sustained_voices_do_not_beat_or_crowd_the_bass():
    voices = so._pad_notes(65.41, so.SCALES["pentatonic"])
    bass = voices[voices < so.BASS_TOP]
    assert {round(float(v / 65.41), 3) for v in bass} <= {1, 1.5, 2, 3}  # roots and fifths

    partials = np.sort(np.concatenate([v * np.arange(1, 6) for v in voices]))
    gaps = np.diff(partials)
    # overtones of different voices either coincide or sit too far apart to beat slowly
    assert not np.any((gaps > 0.5) & (gaps < 8))


def test_duet_is_consonant_only_while_the_variables_covary():
    *_, lead, follower = duet_lines(correlated(0.8))
    _, low, high, *_ = duet_lines(correlated(0.8))
    assert consonant_share(low, high) > 0.97
    _, low, high, *_ = duet_lines(correlated(0.0))
    assert consonant_share(low, high) < 0.9


def test_duet_motion_follows_the_sign_of_the_covariance():
    *_, lead, follower = duet_lines(correlated(0.8))
    assert parallel_share(lead, follower) > 0.75
    *_, lead, follower = duet_lines(correlated(-0.8))
    assert parallel_share(lead, follower) < 0.25


def test_duet_does_not_collapse_into_octaves():
    _, low, high, *_ = duet_lines(correlated(0.7))
    intervals = {round(so._harmonic_height(a, b), 2) for a, b in zip(low, high)}
    assert len(intervals) >= 3


def test_the_correlation_window_can_be_set_in_data_time():
    table = correlated(0.3)
    source = so._load(table, "a*b", 20.0, "db6")
    short = so._score(source, 8.0, correlation_window=20)["correlation"]
    long = so._score(source, 8.0, correlation_window=400)["correlation"]
    assert np.std(np.diff(long)) < np.std(np.diff(short))  # a longer window is smoother
    assert np.abs(long.mean() - 0.3) < 0.15
    default = so._score(source, 8.0)["correlation"]
    n_seconds = table["a"].size / 20.0
    np.testing.assert_allclose(so._score(source, 8.0, so.CORRELATION_SECONDS * n_seconds / 8.0)["correlation"],
                               default)


def test_voices_open_only_while_the_variables_covary():
    # Nothing but the bass sounds below 120 Hz, and the bass follows the flux.
    strong = so.sonify(correlated(0.8), "a*b", duration=6, sample_rate=RATE)
    none = so.sonify(correlated(0.0), "a*b", duration=6, sample_rate=RATE)
    assert share(none.values, 40, 120) < 0.2 * share(strong.values, 40, 120)


@pytest.mark.parametrize("r, heavier, lighter", [(0.8, (60, 70), (50, 59)),
                                                 (-0.8, (50, 59), (60, 70))])
def test_the_bass_names_the_direction_of_the_flux(r, heavier, lighter):
    audio = so.sonify(correlated(r), "a*b", duration=6, sample_rate=RATE)
    assert share(audio.values, *heavier) > 3 * share(audio.values, *lighter)


def test_instruments_default_to_flute_and_bell():
    duet = so.sonify(correlated(0.5), "a*b", duration=1, sample_rate=RATE)
    assert duet.attrs["instruments"] == ["a: flute", "b: bell"]
    solo = so.sonify(pd.Series(red_noise(), name="co2"), duration=1, sample_rate=RATE)
    assert solo.attrs["instruments"] == ["co2: bell"]


def test_instruments_can_be_chosen_per_variable():
    table = correlated(0.5)
    kwargs = dict(duration=2, sample_rate=RATE)
    picked = so.sonify(table, "a*b", instruments=("cello", "marimba"), **kwargs)
    swapped = so.sonify(table, "a*b", instruments=["marimba", "cello"], **kwargs)
    assert picked.attrs["instruments"] == ["a: cello", "b: marimba"]
    assert swapped.attrs["instruments"] == ["a: marimba", "b: cello"]
    assert not np.allclose(picked.values, swapped.values)
    assert so.sonify(red_noise(), instruments="harp", **kwargs).attrs["instruments"][0].endswith("harp")


@pytest.mark.parametrize("name", list(so.INSTRUMENTS))
def test_every_instrument_plays_real_data_cleanly(name):
    audio = so.sonify(red_noise(), instruments=name, duration=3, sample_rate=RATE)
    assert np.isfinite(audio.values).all()
    assert so.integrated_loudness(audio.values, RATE) == pytest.approx(-16, abs=0.5)
    assert np.abs(audio.values).max() <= so.PEAK_CEILING + 1e-4


@pytest.mark.parametrize("name", list(so.INSTRUMENTS))
def test_every_note_starts_and_ends_without_a_click(name):
    timbre = so.INSTRUMENTS[name]
    note = so._note(440.0, 0.5, timbre, RATE)
    assert abs(note[0]) < 1e-9 and abs(note[-1]) < 0.01
    assert note.size == int((0.5 + timbre["release"]) * RATE)


def fingerprint(name):
    """How an instrument differs to the ear: brightness, attack, decay and register."""
    note = so._note(440.0, 1.5, so.INSTRUMENTS[name], RATE)
    envelope = np.abs(spsignal.hilbert(note))
    envelope = np.convolve(envelope, np.ones(64) / 64, "same")
    spectrum = np.abs(np.fft.rfft(note)) ** 2
    hz = np.fft.rfftfreq(note.size, 1 / RATE)
    attack = np.argmax(envelope > 0.5 * envelope.max()) / RATE
    tail = envelope[int(0.8 * RATE)] / envelope.max()
    return np.array([np.log2((hz * spectrum).sum() / spectrum.sum()),
                     np.log2(attack + 0.002), 3 * tail, so.INSTRUMENTS[name]["octave"]])


def test_no_two_instruments_are_the_same_in_disguise():
    prints = {name: fingerprint(name) for name in so.INSTRUMENTS}
    names = list(prints)
    closest = min(np.abs(prints[a] - prints[b]).sum()
                  for k, a in enumerate(names) for b in names[k + 1:])
    assert closest > 0.25


@pytest.mark.parametrize("name, seconds", [("flute", so.MAX_HOLD), ("marimba", 0.9)])
def test_held_notes_last_until_the_next_one_and_struck_ones_ring(name, seconds):
    timbre = so.INSTRUMENTS[name]
    stem = so._play([(0.0, 0, 1.0), (5.0, 1, 0.0)], np.array([220.0, 330.0]), timbre, 0.0,
                    int(7 * RATE), 6.0, RATE)
    # the first note, up to where the silent second one begins
    last = np.flatnonzero(np.abs(stem[:int(5 * RATE)]).max(axis=1) > 1e-6)[-1] / RATE
    assert last == pytest.approx(seconds + timbre["release"], abs=0.02)


def test_an_instrument_sits_in_its_own_register():
    events = [(0.0, 0, 1.0)]
    notes = np.array([440.0])

    def pitch(name):
        stem = so._play(events, notes, so.INSTRUMENTS[name], 0.0, RATE * 3, 3.0, RATE)
        spectrum = np.abs(np.fft.rfft(stem.mean(axis=1)))
        return np.fft.rfftfreq(stem.shape[0], 1 / RATE)[spectrum.argmax()]

    assert pitch("harp") == pytest.approx(440, abs=2)
    assert pitch("glockenspiel") == pytest.approx(880, abs=2)
    assert pitch("pizzicato") == pytest.approx(220, abs=2)


def test_partials_above_nyquist_do_not_fold_back():
    rate = 8000
    note = so._note(1500.0, 0.5, so.INSTRUMENTS["marimba"], rate)
    spectrum = np.abs(np.fft.rfft(note)) ** 2
    hz = np.fft.rfftfreq(note.size, 1 / rate)
    assert spectrum[(hz > 1300) & (hz < 1700)].sum() > 0.99 * spectrum.sum()


def test_a_short_series_plays_its_own_values():
    # A day of half-hourly data: one slot per sample, and the tune rises and
    # falls with the data, each note where its sample is.
    values = np.sin(2 * np.pi * np.arange(48) / 48)
    source = so._load(values, None, None, "db6")
    source["bands"], source["freqs"] = so._decompose(source["values"], source["fs"], "db6")
    score = so._score(source, 30.0)
    slots, grid = so._note_grid(48, score["gate"].size, 30.0)
    assert slots.size == 48
    assert np.corrcoef(score["value"][grid], values)[0, 1] > 0.99


def test_a_long_series_keeps_a_steady_pace():
    slots, _ = so._note_grid(10 ** 6, 6000, 30.0)
    assert slots.size == 30 * so.NOTE_RATE


# --------------------------------------------------------------------------- #
# Mastering
# --------------------------------------------------------------------------- #
def test_k_weighting_reproduces_the_coefficients_of_the_standard():
    shelf, high_pass = so._k_weighting(48000)
    np.testing.assert_allclose(shelf, [1.53512485958697, -2.69169618940638, 1.19839281085285,
                                       1.0, -1.69065929318241, 0.73248077421585], rtol=1e-8)
    np.testing.assert_allclose(high_pass, [1.0, -2.0, 1.0,
                                           1.0, -1.99004745483398, 0.99007225036621], rtol=1e-8)


@pytest.mark.parametrize("rate", [44100, 48000])
def test_loudness_meets_the_reference_of_the_standard(rate):
    # BS.1770: a full-scale 997 Hz tone in one channel reads -3.01 LKFS.
    tone = np.sin(2 * np.pi * 997 * np.arange(5 * rate) / rate)
    assert so.integrated_loudness(tone, rate) == pytest.approx(-3.01, abs=0.02)


def test_loudness_of_silence_is_minus_infinity():
    assert so.integrated_loudness(np.zeros((RATE, 2)), RATE) == -np.inf


@pytest.mark.parametrize("level", [1, 2])
def test_loudness_and_peaks_are_mastered(level):
    audio = so.sonify(red_noise(), level=level, duration=6, loudness=-20, sample_rate=RATE)
    assert so.integrated_loudness(audio.values, RATE) == pytest.approx(-20, abs=0.5)
    assert np.abs(audio.values).max() <= so.PEAK_CEILING + 1e-4
    assert np.abs(audio.values[0]).max() < 1e-3 and np.abs(audio.values[-1]).max() < 1e-3


# --------------------------------------------------------------------------- #
# Quiet passages stay softer, but audible
# --------------------------------------------------------------------------- #
def noise_at(*levels_db, seconds=6.0):
    """Stereo noise holding each level in turn; ``None`` is a gap."""
    rng = np.random.default_rng(0)
    n = int(seconds * RATE)
    return np.concatenate([np.zeros((n, 2)) if db is None
                           else rng.standard_normal((n, 2)) * 10 ** (db / 20)
                           for db in levels_db])


def db(audio, start, stop):
    return 20 * np.log10(rms(audio, start, stop))


def narrowed(audio, n_record=None):
    gain = so._swing_gain(audio, RATE, n_record or audio.shape[0], 6.0)
    return audio if gain is None else audio * gain[:, None]


def test_a_wide_swing_is_narrowed_without_closing():
    audio = noise_at(-30, -10)
    level = narrowed(audio)
    assert db(audio, 8, 10) - db(audio, 2, 4) == pytest.approx(20, abs=0.5)
    assert db(level, 8, 10) - db(level, 2, 4) == pytest.approx(6, abs=1)


def test_soft_never_overtakes_loud():
    level = narrowed(noise_at(-40, -30, -20, -10))
    levels = [db(level, start, start + 2) for start in (2, 8, 14, 20)]
    assert levels == sorted(levels)


def test_a_record_within_the_limit_is_left_alone():
    assert so._swing_gain(noise_at(-12, -10), RATE, int(12 * RATE), 6.0) is None


def test_a_gap_neither_widens_the_range_nor_makes_a_sound():
    assert so._swing_gain(noise_at(-14, None, -10), RATE, int(18 * RATE), 6.0) is None
    level = narrowed(noise_at(-30, None, -10))
    assert not level[int(6.5 * RATE):int(11.5 * RATE)].any()
    assert db(level, 14, 16) - db(level, 2, 4) == pytest.approx(6, abs=1)


def test_sound_beside_a_gap_is_not_boosted():
    level = narrowed(noise_at(-30, -10, None, -10))
    assert db(level, 11, 11.9) == pytest.approx(db(level, 8, 10), abs=1)


def test_the_ring_out_has_no_say():
    assert so._swing_gain(noise_at(-12, -10, -60), RATE, int(12 * RATE), 6.0) is None


@pytest.mark.parametrize("kept, low, high", [(6.0, 3, 8), (None, 20, 40)])
def test_a_calm_half_stays_softer_by_as_much_as_asked(kept, low, high):
    signal = red_noise(slope=-1)
    signal[:signal.size // 2] *= 0.03
    audio = so.sonify(signal, level=1, duration=14, loudness_range=kept, sample_rate=RATE).values

    def loudness(start, stop):
        return so.integrated_loudness(audio[int(start * RATE):int(stop * RATE)], RATE)

    assert low < loudness(9, 12) - loudness(2, 5) < high


# --------------------------------------------------------------------------- #
# Stems: one track per part, which sum to the mix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("level, variable", [(0, "a"), (1, "a"), (2, "a"), (2, "a*b")])
def test_stems_sum_to_the_mix(level, variable):
    kwargs = dict(variable=variable, level=level, duration=4, sample_rate=RATE)
    table = correlated(0.6)
    mix = so.sonify(table, **kwargs).values
    stems = so.sonify(table, stems=True, **kwargs)
    assert stems.dims == ("stem", "time", "channel")
    assert stems.shape[1:] == mix.shape
    np.testing.assert_allclose(stems.sum("stem").values, mix, atol=2e-4)


def test_stems_are_named_by_scale():
    stems = so.sonify(correlated(0.6), "a*b", duration=2, fs=20, stems=True, sample_rate=RATE)
    names = list(stems.stem.values)
    assert "10 Hz" in names and "bass" in names
    assert names[-2:] == ["line: a", "line: b"]
    assert list(so.sonify(red_noise(), level=0, duration=2, fs=20, stems=True, sample_rate=RATE).stem.values)[0] == "trend"


def test_a_stem_holds_only_its_own_scale():
    tone = np.sin(2 * np.pi * BAND_CENTRE * np.arange(2 ** 14))
    stems = so.sonify(tone, level=1, duration=4, fs=1.0, stems=True, sample_rate=RATE)
    levels = (stems ** 2).mean(("time", "channel")).values
    assert stems.stem.values[levels.argmax()] == "0.125 Hz"


# --------------------------------------------------------------------------- #
# Any variable
# --------------------------------------------------------------------------- #
def test_accepts_any_container():
    values = red_noise(2 ** 11)
    series = pd.Series(values, name="co2")
    reference = so.sonify(values, level=0).values
    for data, variable in [(values.tolist(), None), (series, None),
                           (xr.DataArray(values, dims="t"), None),
                           ({"co2": values}, "co2"), (series.to_frame(), "co2"),
                           (series.to_xarray().to_dataset(), "co2")]:
        np.testing.assert_array_equal(so.sonify(data, variable, level=0).values, reference)


def test_blocks_play_one_after_another():
    values = red_noise(2 ** 11)
    blocks = xr.Dataset({"co2": (("TIMESTAMP", "ns"), values.reshape(4, -1))})
    np.testing.assert_array_equal(so.sonify(blocks, "co2", level=0).values,
                                  so.sonify(values, level=0).values)


def test_infers_the_sampling_rate_from_time():
    index = pd.date_range("2022-05-13", periods=2 ** 10, freq="50ms")
    audio = so.sonify(pd.Series(red_noise(2 ** 10), index=index), level=0)
    assert audio.attrs["fs"] == pytest.approx(20)
    assert so.sonify(red_noise(2 ** 10), level=0).attrs["fs"] == 1.0


def test_product_is_the_wavelet_flux_density():
    table = correlated(0.5, slope=0)  # flat, so that the scales hold all the variance
    source = so._load(table, "a*b", 1.0, "db6")
    # summed over scales and time it recovers the covariance
    assert source["values"].mean() == pytest.approx(np.cov(table["a"], table["b"])[0, 1], rel=0.05)
    assert so.sonify(table, "a*b", level=0).attrs["variable"] == "a*b"
    assert so.sonify(table, "a*b", level=1, duration=1, sample_rate=RATE).attrs["level"] == 1


@pytest.mark.filterwarnings("default::RuntimeWarning")  # data_run is not under test here
def test_sonifies_the_output_of_data_run():
    rng = np.random.default_rng(0)
    data = xr.Dataset({name: (("TIMESTAMP", "ns"), rng.standard_normal((2, 2 ** 10)))
                       for name in ("w", "co2")},
                      coords={"TIMESTAMP": pd.date_range("2022-05-13", periods=2, freq="30min"),
                              "ns": np.arange(2 ** 10)})
    result = waveletec.data_run(data, ["w*co2"], fs=20, method="dwt", wavelet="db6", level=5)
    audio = so.sonify(result, "wco2", duration=1, sample_rate=RATE)
    assert audio.attrs["fs"] == 20  # read from the finest band
    assert np.isfinite(audio.values).all()


def test_rejects_bands_of_a_continuous_transform():
    # universal_wt labels those with log2 scales, which would invert the mapping
    bands = xr.DataArray(np.ones((3, 64)), dims=("natural_frequency", "t"),
                         coords={"natural_frequency": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="discrete"):
        so.sonify(bands)


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("n", [4, 5, 7, 16, 48])
def test_short_series_still_make_sound(level, n):
    audio = so.sonify(red_noise(64)[:n], level=level, duration=1, sample_rate=RATE)
    assert np.isfinite(audio.values).all()
    assert np.abs(audio.values).max() > 0


@pytest.mark.parametrize("level", [0, 1, 2])
def test_a_constant_is_silent(level):
    audio = so.sonify(np.full(256, 400.0), level=level, duration=1, sample_rate=RATE)
    assert not audio.values.any()


def test_infinite_values_count_as_gaps():
    signal = red_noise(2 ** 10)
    signal[100] = np.inf
    assert np.isfinite(so.sonify(signal, level=0).values).all()


@pytest.mark.parametrize("call, message", [
    (lambda: so.sonify(np.full(8, np.nan)), "finite"),
    (lambda: so.sonify(np.arange(3.0)), "at least 4"),
    (lambda: so.sonify(red_noise(64), level=3), "level"),
    (lambda: so.sonify(red_noise(64), duration=0), "duration"),
    (lambda: so.sonify(red_noise(64), loudness_range=0), "loudness_range"),
    (lambda: so.sonify(red_noise(64), correlation_window=0), "correlation_window"),
    (lambda: so.sonify(red_noise(64), scale="blues"), "scale"),
    (lambda: so.sonify(red_noise(64), instruments="kazoo"), "instrument"),
    (lambda: so.sonify(red_noise(64), instruments=("harp", "cello")), "takes 1"),
    (lambda: so.sonify(correlated(0.5, 64), "a*b", instruments="harp"), "takes 2"),
    (lambda: so.sonify({"a": red_noise(64)}, "b"), "not found"),
    (lambda: so.sonify({"a": red_noise(64)}), "required"),
])
def test_bad_input_is_explained(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_levels_can_be_written_as_labels():
    signal = red_noise(2 ** 10)
    np.testing.assert_array_equal(so.sonify(signal, level="L0").values,
                                  so.sonify(signal, level=0).values)


def test_accepts_a_custom_scale():
    audio = so.sonify(red_noise(), scale=(1, 3 / 2), duration=1, sample_rate=RATE)
    assert audio.attrs["level"] == 2


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def test_wav_round_trip_creates_folders(tmp_path):
    path = tmp_path / "deep" / "er" / "co2.wav"
    audio = so.sonify(red_noise(2 ** 12), level=1, duration=1, sample_rate=RATE, dst=str(path))
    rate, pcm = wavfile.read(path)
    assert rate == RATE
    assert pcm.dtype == np.int16
    assert pcm.shape == audio.shape
    assert np.abs(pcm / 32767 - audio.values).max() < 1e-3


def test_silence_stays_digital_silence(tmp_path):
    path = tmp_path / "silence.wav"
    so.write_wav(np.zeros((100, 2)), path, sample_rate=RATE)
    assert not wavfile.read(path)[1].any()


def test_is_part_of_the_public_api():
    assert fluxsound.sonify is so.sonify
    assert fluxsound.export_midi is midi.export_midi
    assert {"sonify", "export_midi"} <= set(fluxsound.__all__)


def test_command_line(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for stamp in ("20220513-1200", "20220513-1230"):
        rows = "\n".join(f"{a:.5f}  {b:.5f}" for a, b in rng.standard_normal((512, 2)))
        (tmp_path / f"{stamp}_raw.txt").write_text("header\n" * 9 + "w  co2\n" + rows + "\n")
    wav = tmp_path / "flux.wav"
    monkeypatch.setattr(sys, "argv", [
        "fluxsound", "-i", str(tmp_path / "*_raw.txt"), "-v", "w*co2", "-o", str(wav),
        "-d", "1", "-af", "20", "--scale", "minor_pentatonic", "--sample_rate", str(RATE),
        "--instruments", "cello", "marimba"])
    cli.main()
    rate, pcm = wavfile.read(wav)
    assert rate == RATE and pcm.any()


def test_command_line_explains_a_pattern_that_matches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fluxsound", "-i", str(tmp_path / "nope*"),
                                      "-v", "co2", "-o", str(tmp_path / "x.wav")])
    with pytest.raises(FileNotFoundError, match="No file"):
        cli.main()


# --------------------------------------------------------------------------- #
# MIDI: the same composition, for real instruments
# --------------------------------------------------------------------------- #
TICKS_PER_SECOND = 720  # at 90 BPM and 480 ticks per beat, a slot of 1/3 s is 240 ticks


def key_of(hz):
    return int(round(69 + 12 * np.log2(hz / 440)))


def notes_of(track):
    """``(start, stop, key, velocity)`` of each note of a track, in ticks, in order."""
    now, sounding, notes = 0, {}, []
    for message in track:
        now += message.time
        if message.type == "note_on":
            assert message.note not in sounding  # a key cannot sound twice
            sounding[message.note] = now, message.velocity
        elif message.type == "note_off":
            start, velocity = sounding.pop(message.note)
            notes.append((start, now, message.note, velocity))
    assert not sounding
    return sorted(notes)


def exported(tmp_path, data, variable=None, **kwargs):
    """Write a file and read it back: the file, and the notes of each part by name."""
    mido = pytest.importorskip("mido")
    path = tmp_path / "piece.mid"
    fluxsound.export_midi(data, variable, path, **kwargs)
    read = mido.MidiFile(path, charset="utf-8")
    return read, {track.name: notes_of(track) for track in read.tracks[1:]}


def weight(notes):
    """How much of a part sounds: its notes, each by length and velocity."""
    return sum((stop - start) * velocity for start, stop, _, velocity in notes)


def test_midi_key_of_a_frequency():
    assert midi._key(440.0) == 69
    assert midi._key(65.41) == 36  # the default root, C2
    assert midi._key(65.41 * 5 / 4) == 40  # a pure third is 14 cents flat of the key
    assert midi._key(1.0) == 0 and midi._key(1e5) == 127  # still valid off the keyboard


@pytest.mark.parametrize("variable", ["a", "a*b"])
def test_midi_tracks_are_named_and_ordered_like_the_stems(tmp_path, variable):
    table = correlated(0.6)
    stems = so.sonify(table, variable, duration=2, fs=20, stems=True, sample_rate=RATE)
    read, tracks = exported(tmp_path, table, variable, duration=2, fs=20)
    assert read.type == 1
    assert read.tracks[0].name == variable
    assert list(tracks) == list(stems.stem.values)
    assert all(tracks.values())


def test_midi_line_plays_the_melody(tmp_path):
    signal = red_noise()
    source = so._load(signal, None, None, "db6")
    source["bands"], source["freqs"] = so._decompose(source["values"], source["fs"], "db6")
    score = so._score(source, 10.0)
    slots, grid = so._note_grid(signal.size, score["gate"].size, 10.0)
    notes = so._scale_notes(4 * 65.41, so.SCALES["pentatonic"], 16 * 65.41)
    path = so._line(score["value"][grid], notes.size)
    events = so._events(path, slots, grid, score["gate"], score["activity"])
    assert len(events) > 10

    _, tracks = exported(tmp_path, signal, duration=10)
    assert [(start, key, velocity) for start, _, key, velocity in tracks["line: signal"]] == [
        (round(when * TICKS_PER_SECOND), key_of(notes[index]), round(127 * velocity))
        for when, index, velocity in events]


@pytest.mark.parametrize("correlation_window", [None, 40])
def test_midi_duet_plays_both_paths(tmp_path, correlation_window):
    table = correlated(0.6)
    score, low, high, lead, follower = duet_lines(table, correlation_window=correlation_window)
    slots, grid = so._note_grid(table["a"].size, score["gate"].size, 30.0)
    _, tracks = exported(tmp_path, table, "a*b", fs=1.0, correlation_window=correlation_window)
    for name, path, pitches, part in [("line: a", lead, low, score["parts"][0]),
                                      ("line: b", follower, high, score["parts"][1])]:
        events = so._events(path, slots, grid, score["gate"], part["activity"])
        assert len(events) > 10
        assert [(start, key, velocity) for start, _, key, velocity in tracks[name]] == [
            (round(when * TICKS_PER_SECOND), key_of(pitches[round(when * so.NOTE_RATE)]),
             round(127 * velocity))
            for when, _, velocity in events]


def test_midi_takes_the_correlation_window_of_the_audio(tmp_path):
    table = correlated(0.6)
    usual = exported(tmp_path, table, "a*b", fs=1.0)[1]
    assert exported(tmp_path, table, "a*b", fs=1.0, correlation_window=40)[1] != usual


def test_midi_lines_keep_the_register_of_the_instrument(tmp_path):
    signal = red_noise()
    _, flute = exported(tmp_path, signal, duration=5, instruments="flute")
    _, cello = exported(tmp_path, signal, duration=5, instruments="cello")
    assert ([note[2] - 12 for note in flute["line: signal"]]
            == [note[2] for note in cello["line: signal"]])


def test_midi_notes_last_as_long_as_in_the_audio(tmp_path):
    signal = red_noise()
    _, held = exported(tmp_path, signal, duration=8, instruments="flute")
    notes = held["line: signal"]
    longest = so.MAX_HOLD * TICKS_PER_SECOND
    for (start, stop, *_), (following, *_) in zip(notes, notes[1:]):
        assert stop == min(following, start + longest)

    _, struck = exported(tmp_path, signal, duration=8, instruments="marimba")
    ring = so.INSTRUMENTS["marimba"]["ring"] * TICKS_PER_SECOND
    notes = struck["line: signal"]
    assert max(stop - start for start, stop, *_ in notes) == ring
    for k, (start, stop, key, _) in enumerate(notes):  # unless the same key is played again
        again = [other[0] for other in notes[k + 1:] if other[2] == key]
        assert stop == min(again + [start + ring])


def test_midi_lines_are_what_the_stems_play(tmp_path):
    signal = red_noise()
    stem = so.sonify(signal, instruments="flute", duration=8, stems=True, sample_rate=RATE)
    audio = stem.sel(stem="line: signal").values.mean(axis=1)
    _, tracks = exported(tmp_path, signal, duration=8, instruments="flute")
    hits = []
    for start, stop, key, _ in tracks["line: signal"]:
        if stop - start < 240:  # not enough of a note to tell its pitch
            continue
        a, b = (int(tick / TICKS_PER_SECOND * RATE) for tick in (start, stop))
        spectrum = np.abs(np.fft.rfft(audio[a:b] * np.hanning(b - a), 2 ** 16))
        peak = np.fft.rfftfreq(2 ** 16, 1 / RATE)[spectrum.argmax()]
        hits.append(key_of(peak) == key)
    assert len(hits) > 5 and np.mean(hits) > 0.9


@pytest.mark.parametrize("variable", ["a", "a*b"])
def test_midi_scales_stay_on_the_voices_of_the_audio(tmp_path, variable):
    table = correlated(0.8)
    kwargs = dict(variable=variable, duration=4, root=55.0)
    voices = so.sonify(table, sample_rate=RATE, **kwargs).attrs["voice_frequency_hz"]
    _, tracks = exported(tmp_path, table, **kwargs)
    for notes, voice in zip(tracks.values(), voices):
        assert {key for _, _, key, _ in notes} == {key_of(voice)}


def test_midi_velocities_follow_the_energy_of_a_scale(tmp_path):
    n, duration = 2 ** 14, 8
    t = np.arange(n)
    signal = 0.02 * red_noise(n)
    during = (t > 0.4 * n) & (t < 0.6 * n)
    signal[during] += np.sin(2 * np.pi * BAND_CENTRE * t[during])
    _, tracks = exported(tmp_path, signal, duration=duration, fs=1.0)

    def velocity(notes, second):
        return max((v for start, stop, _, v in notes
                    if start <= second * TICKS_PER_SECOND < stop), default=0)

    scale = tracks["0.125 Hz"]  # the band that holds the tone
    assert velocity(scale, 4) > 3 * max(velocity(scale, 2), velocity(scale, 6))


def test_midi_scale_holds_while_steady_and_sounds_again_as_it_moves():
    loudness = np.concatenate([np.full(600, 0.2), np.linspace(0.2, 1, 600)])  # 3 s each
    slots = np.arange(18) / 3
    notes = midi._sustain(loudness, 60, np.ones(1200), slots, (slots * 200).astype(int), 6.0)
    assert notes[0][0] == 0 and notes[0][1] > 3  # one note for all that is steady
    assert len(notes) >= 3 and notes[-1][1] == 6
    levels = [note[3] for note in notes]
    assert levels == sorted(levels)
    assert all(first[1] == second[0] for first, second in zip(notes, notes[1:]))  # legato


def test_midi_scale_swells_at_half_the_pace_of_the_melody():
    loudness = np.tile([0.2, 0.9, 0.9, 0.2], 8)  # moving at every slot
    grid = np.arange(loudness.size)
    notes = midi._sustain(loudness, 60, np.ones(loudness.size), grid / 3, grid, loudness.size / 3)
    assert len(notes) == loudness.size / midi.VOICE_PACE
    lengths = [stop - start for start, stop, *_ in notes]
    assert lengths == pytest.approx([midi.VOICE_PACE / 3] * len(notes))


def test_midi_a_note_of_a_scale_has_its_mean_loudness():
    loudness = np.linspace(0.2, 1, 1200)  # over 6 s
    slots = np.arange(18) / 3
    notes = midi._sustain(loudness, 60, np.ones(1200), slots, (slots * 200).astype(int), 6.0)
    assert len(notes) > 2
    for start, stop, _, level in notes:
        assert level == pytest.approx(0.2 + 0.8 * (start + stop) / 2 / 6, abs=0.01)


def test_midi_scale_stops_for_a_gap_without_waiting_for_its_pace():
    gate = np.ones(1000)
    gate[300:600] = 0  # from 1.5 s to 3 s
    slots = np.arange(15) / 3
    notes = midi._sustain(np.full(1000, 0.5), 60, gate, slots, (slots * 200).astype(int), 5.0)
    assert [note[:2] for note in notes] == [(0, pytest.approx(5 / 3)), (pytest.approx(10 / 3), 5)]


def test_midi_quiet_scales_rest():
    loudness = np.full(1000, midi.VOICE_FLOOR / 2)
    grid = np.arange(0, 1000, 100)
    assert midi._sustain(loudness, 60, np.ones(1000), grid / 200, grid, 5.0) == []


def test_midi_gaps_are_silent(tmp_path):
    signal = red_noise()
    signal[6000:10000] = np.nan  # from 2.9 s to 4.9 s of 8 s
    _, tracks = exported(tmp_path, signal, duration=8)
    inside = 3.3 * TICKS_PER_SECOND, 4.6 * TICKS_PER_SECOND
    for name, notes in tracks.items():
        assert not [note for note in notes if inside[0] < note[0] < inside[1]]  # no note starts
        if not name.startswith("line"):  # and voices stop, while a line may ring into the gap
            assert not [note for note in notes if note[0] < inside[1] and note[1] > inside[0]]
    assert any(note[0] > inside[1] for note in tracks["line: signal"])  # it plays on afterwards


def test_midi_voices_open_only_while_the_variables_covary(tmp_path):
    _, strong = exported(tmp_path, correlated(0.8), "a*b", duration=6)
    _, none = exported(tmp_path, correlated(0.0), "a*b", duration=6)

    def voices(tracks):
        return sum(weight(notes) for name, notes in tracks.items() if not name.startswith("line"))

    assert voices(none) < 0.2 * voices(strong)


@pytest.mark.parametrize("r, heavier, lighter", [(0.8, 36, 33), (-0.8, 33, 36)])
def test_midi_bass_names_the_direction_of_the_flux(tmp_path, r, heavier, lighter):
    _, tracks = exported(tmp_path, correlated(r), "a*b", duration=6)
    by_key = {key: weight([note for note in tracks["bass"] if note[2] == key])
              for key in (heavier, lighter)}
    assert by_key[heavier] > 3 * by_key[lighter]


def test_midi_tempo_keeps_the_clock_of_the_audio(tmp_path):
    mido = pytest.importorskip("mido")
    read, tracks = exported(tmp_path, red_noise(), duration=10)
    tempo = next(message.tempo for message in read.tracks[0] if message.type == "set_tempo")
    assert mido.tempo2bpm(tempo) == pytest.approx(90)
    assert mido.tick2second(240, read.ticks_per_beat, tempo) == pytest.approx(1 / so.NOTE_RATE)
    assert all(start % 240 == 0 for notes in tracks.values() for start, *_ in notes)  # on the grid
    assert 10 <= read.length < 10 + so.RING_OUT


@pytest.mark.parametrize("n, duration, bpm", [(48, 30, 48), (48, 40, 72), (4, 60, 64)])
def test_midi_slots_of_a_short_series_stay_in_time(tmp_path, n, duration, bpm):
    # One slot per sample. A slot grows into longer notes rather than crawl at 2 BPM.
    mido = pytest.importorskip("mido")
    read, tracks = exported(tmp_path, red_noise(64)[:n], duration=duration)
    tempo = next(message.tempo for message in read.tracks[0] if message.type == "set_tempo")
    assert mido.tempo2bpm(tempo) == pytest.approx(bpm)
    first, second = sorted({start for notes in tracks.values() for start, *_ in notes})[:2]
    assert mido.tick2second(second - first, read.ticks_per_beat, tempo) == pytest.approx(duration / n)


@pytest.mark.parametrize("variable", ["a", "a*b"])
def test_midi_has_a_channel_for_each_part_even_with_many_scales(tmp_path, variable):
    table = correlated(0.5, 2 ** 19)  # 15 scales, which with a line is more than there are channels
    read, _ = exported(tmp_path, table, variable, duration=2)
    assert len(read.tracks) - 1 > 15
    channels = {track.name: track[1].channel for track in read.tracks[1:]}
    assert midi.DRUM_CHANNEL not in channels.values()
    own = [channel for name, channel in channels.items() if not name.endswith("Hz")]
    assert len(set(own)) == len(own)
    assert not set(own) & {channel for name, channel in channels.items() if name.endswith("Hz")}


def test_midi_programs_are_those_of_the_instruments(tmp_path):
    read, _ = exported(tmp_path, correlated(0.5), "a*b", duration=2, instruments=("cello", "marimba"))
    programs = {track.name: track[1].program for track in read.tracks[1:]}
    assert programs["line: a"] == 42 and programs["line: b"] == 12
    assert programs["bass"] == midi.BASS_PROGRAM


def test_every_instrument_has_its_own_midi_program():
    assert set(midi.PROGRAMS) == set(so.INSTRUMENTS)
    assert len(set(midi.PROGRAMS.values())) == len(midi.PROGRAMS)
    assert all(0 <= program < 128 for program in midi.PROGRAMS.values())


@pytest.mark.parametrize("scale", [(1, 1.001, 3 / 2), (1, 3 / 2, 2)])
def test_a_key_never_sounds_twice(tmp_path, scale):
    # Near-equal degrees share a key, and in the second scale so do both bass notes.
    exported(tmp_path, correlated(0.5), "a*b", duration=8, scale=scale, instruments=("bell", "harp"))


@pytest.mark.parametrize("variable", ["a", "a*b"])
def test_midi_depends_on_the_data_only(tmp_path, variable):
    table = correlated(0.5)
    first = exported(tmp_path, table, variable, duration=4)[1]
    assert first == exported(tmp_path, table, variable, duration=4)[1]
    assert first != exported(tmp_path, correlated(0.5, slope=-1), variable, duration=4)[1]


@pytest.mark.parametrize("n", [4, 5, 7, 16, 48])
def test_short_series_still_make_midi(tmp_path, n):
    _, tracks = exported(tmp_path, red_noise(64)[:n], duration=1)
    assert tracks["line: signal"]


def test_midi_of_a_constant_is_silent(tmp_path):
    _, tracks = exported(tmp_path, np.full(256, 400.0), duration=1)
    assert len(tracks) > 1 and not any(tracks.values())


def test_midi_accepts_the_output_of_data_run_and_any_name(tmp_path):
    bands = xr.DataArray(red_noise(2 ** 10).reshape(4, -1), dims=("natural_frequency", "t"),
                         coords={"natural_frequency": [8.0, 4.0, 2.0, 1.0]})
    read, tracks = exported(tmp_path, {"CO₂": bands}, "CO₂", duration=2)
    assert read.tracks[0].name == "CO₂" and "line: CO₂" in tracks and "8 Hz" in tracks


def test_midi_creates_folders_and_returns_the_file(tmp_path):
    mido = pytest.importorskip("mido")
    path = tmp_path / "deep" / "er" / "co2.mid"
    written = fluxsound.export_midi(red_noise(2 ** 10), None, str(path), duration=1)
    assert isinstance(written, mido.MidiFile)
    assert len(mido.MidiFile(path).tracks) == len(written.tracks)


@pytest.mark.parametrize("kwargs, message", [
    (dict(duration=0), "duration"),
    (dict(correlation_window=0), "correlation_window"),
    (dict(scale="blues"), "scale"),
    (dict(instruments="kazoo"), "instrument"),
    (dict(instruments=("harp", "cello")), "takes 1"),
])
def test_bad_input_to_midi_is_explained(tmp_path, kwargs, message):
    pytest.importorskip("mido")
    with pytest.raises(ValueError, match=message):
        fluxsound.export_midi(red_noise(64), None, tmp_path / "x.mid", **kwargs)
    assert not (tmp_path / "x.mid").exists()


def test_midi_explains_how_to_get_mido(tmp_path, monkeypatch):
    monkeypatch.setattr(midi, "mido", None)
    with pytest.raises(ImportError, match=r"fluxsound\[midi\]"):
        fluxsound.export_midi(red_noise(64), None, tmp_path / "x.mid")


def command_line_input(tmp_path):
    rng = np.random.default_rng(0)
    rows = "\n".join(f"{a:.5f}  {b:.5f}" for a, b in rng.standard_normal((512, 2)))
    (tmp_path / "20220513-1200_raw.txt").write_text("header\n" * 9 + "w  co2\n" + rows + "\n")
    return str(tmp_path / "*_raw.txt")


def test_command_line_writes_midi(tmp_path, monkeypatch):
    mido = pytest.importorskip("mido")
    path = tmp_path / "flux.mid"
    monkeypatch.setattr(sys, "argv", [
        "fluxsound", "-i", command_line_input(tmp_path), "-v", "w*co2", "-o", str(path),
        "-d", "2", "-af", "20", "--scale", "minor_pentatonic", "--instruments", "cello", "marimba"])
    cli.main()
    read = mido.MidiFile(path)
    assert read.tracks[0].name == "w*co2"
    assert [track.name for track in read.tracks[-3:]] == ["bass", "line: w", "line: co2"]
    assert "10 Hz" in [track.name for track in read.tracks]
    assert read.tracks[-2][1].program == midi.PROGRAMS["cello"]


def test_command_line_midi_is_level_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fluxsound", "-i", str(tmp_path / "nope*"), "-v", "co2",
                                      "-o", str(tmp_path / "x.mid"), "-l", "1"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "level 2" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Decomposition
# --------------------------------------------------------------------------- #
def test_scales_and_trend_add_up_to_the_signal():
    split = decompose(red_noise(2 ** 12), fs=20)
    np.testing.assert_allclose(split.scales.sum("frequency") + split.trend, split.signal, atol=1e-9)
    assert split.frequency.values[0] == 10
    assert split.share.sum() == pytest.approx(1)


def test_shares_of_a_flux_are_the_cospectrum_and_can_be_negative():
    a = red_noise(2 ** 12, 1, slope=0)
    table = {"a": a, "b": -a}
    split = decompose(table, "a*b", fs=20)
    assert split.attrs["product"]
    assert split.share.sum() == pytest.approx(1)
    assert (split.scales.mean("time") < 0).all()
