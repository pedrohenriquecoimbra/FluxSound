"""A signal split into wavelet scales, ready to plot or to hear.

Both the decomposition app and a notebook call :func:`decompose`, so what is
shown on the page is what you get in Python.
"""
import numpy as np
import xarray as xr

from .sonification import _decompose, _load

__all__ = ["decompose"]


def decompose(data, variable=None, fs=None, wavelet="db6"):
    """Split a variable into wavelet scales with waveletec.

    Args:
        data: Anything :func:`fluxsound.sonify` accepts.
        variable: Name of the variable, or a product such as ``"w*co2"``, which
            gives the cross-scalogram: the flux density at each scale.
        fs: Sampling rate of the data (Hz).
        wavelet: Mother wavelet.

    Returns:
        xarray.Dataset with ``scales (frequency, time)``, finest first, ``signal``
        and ``trend`` over ``time`` (s), and ``share (frequency)``: each scale's
        part of the variance, or of the covariance for a product (it can be
        negative). ``signal`` is the sum of ``scales`` and ``trend``.
    """
    source = _load(data, variable, fs, wavelet)
    product = "parts" in source
    if "bands" in source:
        bands, freqs = source["bands"], source["freqs"]
    else:
        bands, freqs = _decompose(source["values"], source["fs"], wavelet)
    values = source["values"]
    trend = values - bands.sum(axis=0)
    weights = bands.mean(axis=1) if product else bands.var(axis=1)
    total = weights.sum()
    share = weights / total if total else np.zeros_like(weights)
    time = np.arange(values.size) / source["fs"]
    return xr.Dataset(
        {"scales": (("frequency", "time"), bands), "signal": ("time", values),
         "trend": ("time", trend), "share": ("frequency", share)},
        coords={"frequency": freqs, "time": time},
        attrs=dict(variable=source["name"], fs=source["fs"], wavelet=wavelet, product=product))
