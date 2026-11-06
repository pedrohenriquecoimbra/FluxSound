"""A composition as MIDI, to open in a DAW or a score editor.

The instruments of :mod:`fluxsound.sonification` are sketches. A MIDI file keeps
the composition and leaves the sound to real ones. It holds the notes of level 2,
on the clock of the audio and with tracks named like its stems, so that a wavelet
scale can be reorchestrated or rewritten next to the stem it came from.

Melody lines go over as they are. A sustained voice needs a translation: its pitch
never changes and its loudness follows a curve, while a MIDI note has one velocity.
So where a line sounds a note when its pitch has moved, a voice sounds one when its
loudness has moved. A steady scale is one long note, and the velocities of a track
trace the energy at that scale.
"""

# built-in modules
import logging
import os

# 3rd party modules
import numpy as np

try:
    import mido
except ImportError:  # optional, see the `midi` extra
    mido = None

from .sonification import (DEFAULT_DURATION, INSTRUMENTS, LOUDNESS_EXPONENT, _band_names,
                           _bass_notes, _cast, _check_correlation_window, _check_duration,
                           _decompose, _hold, _load, _melody, _note_grid, _pad_voices,
                           _parse_instruments, _parse_scale, _score, _slot_seconds, _unit)

logger = logging.getLogger(__name__)

__all__ = ["export_midi"]

TICKS_PER_BEAT = 480
SLOWEST_BPM = 40             # a melody slot is an eighth note, longer if that would crawl
DRUM_CHANNEL = 9             # holds a drum kit in General MIDI, whatever the program

VOICE_PACE = 2               # melody slots per chance to change: swells, not tremolo
VOICE_STEP = 0.2             # change of loudness (0..1) that sounds a new note
VOICE_FLOOR = 0.1            # loudness (0..1) under which a voice rests

# The closest General MIDI programs, counted from 0, so that a file sounds about
# right as it is opened. In a DAW they only suggest what to load on each track.
PROGRAMS = {
    "flute": 73, "clarinet": 71, "oboe": 68, "bassoon": 70, "saxophone": 65, "horn": 60,
    "trumpet": 56, "trombone": 57, "violin": 40, "viola": 41, "cello": 42, "double_bass": 43,
    "voice": 52, "organ": 19, "accordion": 21, "bell": 14, "piano": 0, "harp": 46,
    "harpsichord": 6, "guitar": 24, "electric_guitar": 27, "bass_guitar": 33, "pizzicato": 45,
    "marimba": 12, "vibraphone": 11, "glockenspiel": 9, "celesta": 8, "timpani": 47,
    "kalimba": 108,
}
VOICE_PROGRAM = 48           # String Ensemble 1
BASS_PROGRAM = 43            # Contrabass


def _key(hz):
    """Nearest MIDI key: keys are equal-tempered, the scales here are pure."""
    return int(np.clip(np.round(69 + 12 * np.log2(hz / 440)), 0, 127))


def _velocity(level):
    """A level in 0..1 as a velocity, which a note-on cannot have at 0."""
    return int(np.clip(np.round(127 * level), 1, 127))


def _line_notes(events, notes, timbre, duration):
    """A melody line as ``(start, stop, key, level)``, times in s, as the audio plays it."""
    return [(when, when + _hold(events, k, timbre, duration),
             _key(notes[index] * 2.0 ** timbre["octave"]), velocity)
            for k, (when, index, velocity) in enumerate(events)]


def _sustain(loudness, key, gate, slots, grid, duration):
    """A sustained voice as ``(start, stop, key, level)``, times in s.

    A note lasts until the loudness has moved ``VOICE_STEP`` away from where it
    began, and then the next one sounds. The voice rests in gaps and below
    ``VOICE_FLOOR``. The level of a note is its mean loudness rather than where it
    began, which would trail behind the curve.
    """
    played = []
    start = first = began = None
    for k, (when, at) in enumerate(zip(slots, grid)):
        if k % VOICE_PACE and gate[at] > 0.5:  # only a gap cannot wait
            continue
        now = loudness[at]
        sounding = gate[at] > 0.5 and now >= VOICE_FLOOR
        if start is not None and (not sounding or abs(now - began) >= VOICE_STEP):
            played.append((start, float(when), key, loudness[first:at].mean()))
            start = None
        if start is None and sounding:
            start, first, began = float(when), at, now
    if start is not None:
        played.append((start, duration, key, loudness[first:].mean()))
    return played


def _voices(source, score, duration, ratios, root):
    """Notes of each sustained voice, coarsest first, then of both notes of a bass."""
    slots, grid = _note_grid(source["values"].size, score["gate"].size, duration)
    curves, pitches = list(score["amp"]), list(_pad_voices(score, ratios, root))
    if "parts" in score:
        level = score["activity"] * score["gate"]
        curves += [level * score["direction"], level * (1 - score["direction"])]
        pitches += _bass_notes(root, ratios)
    # The curves are pressures, and a velocity is closer to a loudness. Harmony comes
    # after the scaling, so that voices open only as far as the covariance, as they
    # do in the audio.
    loudness = (_unit(np.array(curves) ** LOUDNESS_EXPONENT)
                * score.get("harmony", 1.0) ** LOUDNESS_EXPONENT)
    return [_sustain(row, _key(hz), score["gate"], slots, grid, duration)
            for row, hz in zip(loudness, pitches)]


def _parts(source, score, duration, ratios, root, players, chosen):
    """Name, program and notes of every part, in the order of the stems."""
    voices = _voices(source, score, duration, ratios, root)
    parts = [(name, VOICE_PROGRAM, notes) for name, notes in zip(_band_names(source["freqs"]), voices)]
    if "parts" in score:
        parts.append(("bass", BASS_PROGRAM, voices[-2] + voices[-1]))
    lines = _melody(source, score, duration, ratios, root, [INSTRUMENTS[name] for name in chosen])
    for player, name, (events, notes, timbre, _) in zip(players, chosen, lines):
        parts.append((f"line: {player}", PROGRAMS[name], _line_notes(events, notes, timbre, duration)))
    return parts


def _channels(n_parts, n_voices):
    """A channel per part, skipping the drums.

    Each line and the bass get their own, since each has its own program. There
    can be more scales than channels, but all voices share a program, so they can
    share channels too.
    """
    free = [channel for channel in range(16) if channel != DRUM_CHANNEL]
    own, shared = free[:n_parts - n_voices], free[n_parts - n_voices:]
    return [shared[j % len(shared)] for j in range(n_voices)] + own


def _track(name, program, channel, notes, ticks_per_second):
    """A named MIDI track from ``(start, stop, key, level)`` notes."""
    played = sorted((int(round(start * ticks_per_second)), int(round(stop * ticks_per_second)),
                     key, _velocity(level)) for start, stop, key, level in notes)
    messages = []
    again = {}
    for start, stop, key, velocity in reversed(played):
        # A key cannot sound twice, so a ringing note stops where its key is played again.
        stop = min(stop, again.get(key, stop))
        again[key] = start
        if stop > start:
            messages += [(start, 1, "note_on", key, velocity), (stop, 0, "note_off", key, 0)]

    track = mido.MidiTrack([mido.MetaMessage("track_name", name=name),
                            mido.Message("program_change", channel=channel, program=program)])
    now = 0
    for tick, _, kind, key, velocity in sorted(messages):  # offs first, for a key played again
        track.append(mido.Message(kind, channel=channel, note=key, velocity=velocity,
                                  time=tick - now))
        now = tick
    return track


def export_midi(data, variable, dst, duration=None, fs=None, scale="pentatonic", root=65.41,
                instruments=None, wavelet="db6", correlation_window=None):
    """Write the music of level 2 as a MIDI file, to play with real instruments.

    There is a track per wavelet scale and per melody line, plus the bass of a
    duet, in the order and with the names of ``sonify(..., stems=True)``. The same
    arguments give the same composition, so stems and tracks line up from time 0.

    - **Melody lines** hold the notes of the audio, with the same length and in the
      register of the instrument. Velocities come from the activity of the variable.
    - **Scales** stay on the note of their voice, and sound again whenever their
      loudness has moved (see the module docstring). Velocities follow the energy
      at that scale, and for a duet they open only as far as the covariance.

    A melody slot is an eighth note, which at the usual pace makes 90 BPM, so scores
    stay readable. Keep that tempo to stay in time with the stems.

    Only the notes are kept. Keys are equal-tempered, so the pure ratios of the audio
    become the nearest key, and pan, room and mastering stay in the stems.

    Args:
        data: Anything :func:`fluxsound.sonify` accepts.
        variable: Name of the variable, or a product such as ``"w*co2"``, played as
            a duet. ``None`` if *data* is a single series.
        dst: Path of the ``.mid`` file to write; missing folders are created.
        duration: Seconds of music for the whole record (default 30).
        fs: Sampling rate of the data (Hz). Only labels the scales.
        scale: A name in ``SCALES`` or a tuple of frequency ratios.
        root: Lowest note in Hz (default C2).
        instruments: A name in ``INSTRUMENTS``, or one per variable of a duet, in
            the order of the product. Sets the General MIDI program of each line
            and its register. By default ``"bell"``, and ``("flute", "bell")`` for
            a duet.
        wavelet: Mother wavelet for the decomposition.
        correlation_window: Duet only. Seconds of data over which the local
            correlation is judged, as in :func:`fluxsound.sonify`.

    Returns:
        The ``mido.MidiFile`` that was written. Its first track holds the name of
        the variable and the tempo; the parts follow.

    Raises:
        ImportError: If mido is not installed.
        ValueError: If the scale, instrument or variable is unknown, the number of
            instruments is wrong, or there are not enough finite values.
    """
    if mido is None:
        raise ImportError('MIDI export needs mido: pip install "fluxsound[midi]"')
    _check_duration(duration)
    _check_correlation_window(correlation_window)
    ratios = _parse_scale(scale)
    chosen = _parse_instruments(instruments)
    duration = DEFAULT_DURATION if duration is None else duration

    source = _load(data, variable, fs, wavelet)
    if "bands" not in source:
        source["bands"], source["freqs"] = _decompose(source["values"], source["fs"], wavelet)
    players, chosen = _cast(source, chosen)
    score = _score(source, duration, correlation_window)
    parts = _parts(source, score, duration, ratios, root, players, chosen)
    if np.ptp(source["values"]) == 0:
        logger.warning("%s is constant: there is nothing to hear.", source["name"])
        parts = [(name, program, []) for name, program, _ in parts]

    slot = _slot_seconds(source["values"].size, duration)
    beats = 0.5
    while 60 * beats / slot < SLOWEST_BPM:
        beats *= 2
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT, charset="utf-8")
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("track_name", name=source["name"]),
        mido.MetaMessage("set_tempo", tempo=int(round(1e6 * slot / beats)))]))
    channels = _channels(len(parts), score["amp"].shape[0])
    for (name, program, notes), channel in zip(parts, channels):
        midi.tracks.append(_track(name, program, channel, notes, TICKS_PER_BEAT * beats / slot))

    if os.path.dirname(str(dst)):
        os.makedirs(os.path.dirname(str(dst)), exist_ok=True)
    midi.save(str(dst))
    logger.info("Saved %s (%d parts, %.4g BPM).", dst, len(parts), 60 * beats / slot)
    return midi
