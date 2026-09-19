"""Load FLAME ``generic_model.pkl`` without the chumpy package."""

from __future__ import annotations

import pickle
import sys
from types import ModuleType

import numpy as np


class _ChumpyArray:
    """Stand-in for chumpy.Ch so FLAME pickles load without the chumpy package."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        if args:
            self.x = args[0]
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state

    def _numpy(self):
        for key in ("x", "r"):
            if key in self.__dict__:
                return np.asarray(self.__dict__[key])
        return np.asarray(0.0)

    def __array__(self, dtype=None):
        return np.asarray(self._numpy(), dtype=dtype)

    @property
    def shape(self):
        return self._numpy().shape

    @property
    def ndim(self):
        return self._numpy().ndim


def _install_chumpy_shim() -> None:
    if "chumpy" in sys.modules:
        return
    chumpy = ModuleType("chumpy")
    chumpy_ch = ModuleType("chumpy.ch")
    chumpy.Ch = _ChumpyArray
    chumpy.ch = chumpy_ch
    chumpy_ch.Ch = _ChumpyArray
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = chumpy_ch


def load_flame_pickle(path) -> dict:
    _install_chumpy_shim()
    with open(path, "rb") as handle:
        return pickle.load(handle, encoding="latin1")


def flame_to_numpy(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)
