# Copyright (c) 2025 The Tremelique Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Wavelets used as sources for the wave simulators.
"""

import copy

import numpy as np


class RickerWavelet:
    def __init__(self, amp, f_cut, delay=0):
        self.amp = amp
        self.f_cut = f_cut
        self.delay = delay

    def copy(self):
        return copy.deepcopy(self)

    def __call__(self, time):
        sqrt_pi = np.sqrt(np.pi)
        fc = self.f_cut / (3 * sqrt_pi)
        # Standard delay to make the wavelet start at time zero and be causal
        td = time - 2 * sqrt_pi / self.f_cut
        # Apply the user defined delay on top
        t = td - self.delay
        scale = self.amp * (2 * np.pi * (np.pi * fc * t) ** 2 - 1)
        res = scale * np.exp(-np.pi * (np.pi * fc * t) ** 2)
        return res
