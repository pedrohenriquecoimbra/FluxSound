# FluxSound

Hear eddy-covariance data. Built on [waveletec](https://github.com/pedrohenriquecoimbra/wavelet-ec).

Two apps, both static pages, so they can be served by any web host:

| App | What it is | Wavelets |
| --- | --- | --- |
| **Ensemble** | Compose with a year of daily ICOS data: orchestra, quartet, band or choir. Every composition is a link. | No: it plays daily values directly |
| **Decomposition** | Half an hour of 20 Hz data split into wavelet scales. Turn scales on and off, see what they add up to, hear the result. | Yes |

## Python

```bash
pip install -e ".[test]"
```

```python
import fluxsound

fluxsound.sonify(data, "co2", level=0, dst="co2_L0.wav")  # the signal as an audio file
fluxsound.sonify(data, "co2", level=1, dst="co2_L1.wav")  # altered for human hearing
fluxsound.sonify(data, "w*co2", dst="flux.wav")           # music: a flux is a duet
fluxsound.sonify(data, "w*co2", instruments=("cello", "marimba"), dst="flux.wav")

stems = fluxsound.sonify(data, "co2", stems=True)         # one track per wavelet scale
fluxsound.export_midi(data, "w*co2", "flux.mid")          # the same music for a DAW

from fluxsound.decomposition import decompose
split = decompose(data, "w*co2", fs=20)                   # what the Decomposition app shows
```

Or from the command line:

```bash
fluxsound -i "level_6/20220513-12*" -v "w*co2" -o flux.wav
fluxsound -i "level_6/20220513-12*" -v "w*co2" -o flux.mid
```

### Levels

- **L0, audio file.** The signal written as audio.
- **L1, audible.** The same signal altered for hearing: each wavelet scale is expanded into the hearing range, equalised and weighted for equal loudness. Every sound is still a piece of the data.
- **L2, music.** Each scale drives a voice on a consonant scale and the value plays a melody. A flux is a duet: one instrument per variable, and their covariance is heard as harmony.

### Working in a DAW

The instruments here are sketches. For real ones, take the L2 composition to Reaper, Ableton or MuseScore as MIDI, with the stems as a reference:

```bash
pip install -e ".[midi]"
```

```python
kwargs = dict(duration=30, scale="pentatonic", instruments=("cello", "marimba"))

fluxsound.export_midi(data, "w*co2", "daw/flux.mid", **kwargs)
stems = fluxsound.sonify(data, "w*co2", stems=True, **kwargs)
for name in stems.stem.values:
    fluxsound.write_wav(stems.sel(stem=name), f"daw/{name.replace(':', '')}.wav")
```

Tracks and stems have the same names and, with the same arguments, the same notes: one per wavelet scale (`0.078125 Hz`), the `bass` of a duet, and one per melody line (`line: co2`). Each line carries the General MIDI program of its instrument, a starting point for loading your own.

- **Lines** are the notes of the audio, with velocities from the activity of the variable.
- **Scales** stay on the note of their voice. A note sounds whenever the energy at that scale has moved, so the velocities trace it: a steady scale is one long note.
- **Time.** A melody slot is an eighth note, which makes 90 BPM unless the record has few samples, and the file carries its tempo. Import it when asked, keep the tracks separate, and place everything at the start of the project: tracks and stems then stay in time.
- **Not in the file.** Keys are equal-tempered, so the pure ratios of the audio become the nearest key. Pan, room and mastering stay in the stems.

## Apps

```bash
pip install -e ".[apps]"
python scripts/build_all.py     # writes dist/
python -m http.server --directory dist
```

The sources are in `apps/`. Pushing to `main` publishes them with GitHub Pages (`.github/workflows/pages.yml`); set Pages > Source to "GitHub Actions" in the repository settings once.

## Data

- Ensemble: daily data downloaded from the ICOS store at build time. ICOS RI et al., 2026, Ecosystem final quality (L2) product in ETC-Archive and associated formats, INTERIM release 2026-2, https://doi.org/10.18160/JX1S-VXZQ, CC BY 4.0.
- Decomposition: `data/fr-gri-2022-05-13-1230.npz`, half an hour of FR-Gri from the waveletec sample, from the ICOS Carbon Portal, CC BY 4.0.
- Land outline: Natural Earth, public domain.

## License

EUPL-1.2.
