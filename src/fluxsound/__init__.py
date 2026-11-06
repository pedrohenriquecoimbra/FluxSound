"""Hear eddy-covariance data. Built on waveletec."""
from .midi import export_midi
from .sonification import INSTRUMENTS, SCALES, integrated_loudness, sonify, write_wav

__version__ = "0.1.0"
__all__ = ["sonify", "export_midi", "write_wav", "integrated_loudness", "INSTRUMENTS", "SCALES"]
