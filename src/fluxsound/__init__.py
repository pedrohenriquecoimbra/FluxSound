"""Hear eddy-covariance data: the sound tools of fluxscale, under a short name."""
from fluxscale import decompose, export_midi, sonify
from fluxscale.extra.sonification import INSTRUMENTS, SCALES, integrated_loudness, write_wav

__version__ = "0.1.0"
__all__ = ["sonify", "decompose", "export_midi", "write_wav", "integrated_loudness", "INSTRUMENTS", "SCALES"]
