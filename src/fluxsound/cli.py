"""Command line: ``fluxsound -i FILES -v VARIABLE -o OUT.wav``, or ``OUT.mid`` for MIDI."""
import argparse
import glob
import logging

import xarray as xr
from waveletec.io import READERS
from waveletec.main import __custom_params__ as extra_arguments

from .midi import export_midi
from .sonification import sonify


def main():
    parser = argparse.ArgumentParser(
        description="Turn a variable into sound, or into MIDI if the output ends in .mid. "
                    "Other `sonify` or `export_midi` arguments can be passed as `--key value` "
                    "(e.g. `--instruments cello marimba`).")
    parser.add_argument("-i", "--input_path", nargs="+", required=True,
                        help='Input file(s), in time order; patterns such as "level_6/20220513*" are expanded')
    parser.add_argument("-v", "--variable", required=True,
                        help='Variable, or a product played as a duet (e.g., "co2", "w*co2")')
    parser.add_argument("-o", "--output", required=True, help="Path of the .wav or .mid file to write")
    parser.add_argument("-l", "--level", type=int, default=2, choices=[0, 1, 2],
                        help="0 audio file, 1 audible, 2 music (default: 2)")
    parser.add_argument("-d", "--duration", type=float, default=None,
                        help="Seconds of sound (default: 30; at level 0, one audio sample per data sample)")
    parser.add_argument("-af", "--acquisition_frequency", type=float, default=None,
                        help="Data acquisition frequency in Hz; only labels the output")
    parser.add_argument("--reader", default="ep_raw_lvl",
                        help='Reader for the input files (default: "ep_raw_lvl")')
    args, unknown = parser.parse_known_args()
    midi = args.output.lower().endswith((".mid", ".midi"))
    if midi and args.level != 2:
        parser.error("MIDI holds the notes of level 2: drop --level or write a .wav file.")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = sorted(p for pattern in args.input_path for p in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No file matches {args.input_path}.")
    data = xr.concat([READERS.get(args.reader)(p) for p in paths], dim="TIMESTAMP")
    if midi:
        export_midi(data, args.variable, args.output, duration=args.duration,
                    fs=args.acquisition_frequency, **extra_arguments(unknown))
    else:
        sonify(data, args.variable, level=args.level, duration=args.duration,
               fs=args.acquisition_frequency, dst=args.output, **extra_arguments(unknown))
