# FluxSound

Hear eddy-covariance data. Built on [FluxScale](https://github.com/pedrohenriquecoimbra/wavelet-ec), which splits fluxes into wavelet scales and turns any variable into sound.

Two apps, both static pages, so they can be served by any web host:

| App | What it is | Wavelets |
| --- | --- | --- |
| **Ensemble** | Compose with a year of daily ICOS data: orchestra, quartet, band or choir. Every composition is a link. | No: it plays daily values directly |
| **Decomposition** | Half an hour of 20 Hz data split into wavelet scales. Turn scales on and off, see what they add up to, hear the result. | Yes |

## Apps

```bash
pip install -r requirements.txt
python scripts/build_all.py     # writes dist/
python -m http.server --directory dist
```

The sources are in `apps/`. Pushing to `main` publishes them with GitHub Pages (`.github/workflows/pages.yml`); set Pages > Source to "GitHub Actions" in the repository settings once.

## Python

The sound tools live in FluxScale. This package gives them a short name and a `fluxsound` command:

```bash
pip install fluxsound
```

```python
import fluxsound

fluxsound.sonify(data, "w*co2", dst="flux.wav")   # a flux as a duet
fluxsound.export_midi(data, "w*co2", "flux.mid")  # the same music for a DAW (pip install "fluxsound[midi]")
```

```bash
fluxsound -i "level_6/20220513-12*" -v "w*co2" -o flux.wav
```

Levels, instruments and working in a DAW are described in the [FluxScale README](https://github.com/pedrohenriquecoimbra/wavelet-ec#readme).

## Data

- Ensemble: daily data downloaded from the ICOS store at build time. ICOS RI et al., 2026, Ecosystem final quality (L2) product in ETC-Archive and associated formats, INTERIM release 2026-2, https://doi.org/10.18160/JX1S-VXZQ, CC BY 4.0.
- Decomposition: `data/fr-gri-2022-05-13-1230.npz`, half an hour of FR-Gri from the FluxScale sample, from the ICOS Carbon Portal, CC BY 4.0.
- Land outline: Natural Earth, public domain.

## License

EUPL-1.2.
