"""Scalar recurrence: exp(A)=1/2, delta=1, B=1, C=2, D=1/4."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from mamba_numpy.scan import selective_scan

u = np.array([[[2.], [3.], [1.], [4.]]])
delta = np.ones_like(u)
A = np.array([[np.log(0.5)]])
B = np.ones_like(u)
C = 2 * np.ones_like(u)
D = np.array([0.25])
y, last, cache = selective_scan(u, delta, A, B, C, D, retain=True)
for t in range(u.shape[1]):
    old, new = cache.states[0, t, 0, 0], cache.states[0, t + 1, 0, 0]
    value = u[0, t, 0]
    print(f"t={t}: h = 0.5*{old:g} + {value:g} = {new:g}; "
          f"y = 2*{new:g} + 0.25*{value:g} = {y[0,t,0]:g}")
