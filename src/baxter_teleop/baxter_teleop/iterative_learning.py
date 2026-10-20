"""Iterative learning control for replays: correct each practice run with the previous run's error.

  Arimoto, Kawamura, Miyazaki, "Bettering operation of robots by learning", J. Robotic Systems 1984.
  Bristow, Tharayil, Alleyne, "A survey of iterative learning control", IEEE Control Systems 2006.

A replay repeats the same motion in the same conditions, so most of its error repeats too (friction,
the joints' springs, the servo's lag). After run k, with e_k the demo minus the measured angles on
the demo's grid, the next run commands the demo plus

    offset_{k+1}(t) = offset_k(t) + gain * F(e_k)(t + lead)

`lead` (the replay's lag) aligns each correction with the command that caused the error, and F is a
zero-phase low-pass filter (a Butterworth run forward and backward), so the correction adds no delay
and does not learn noise above `cutoff`. A gain below 1 converges without overshoot.
"""

import numpy as np
from scipy.signal import butter, filtfilt


class IterativeLearning:
    """The learning update for one arm's command offset on a uniform time grid."""

    def __init__(self, grid, gain=0.8, cutoff=3.0, lead=0.0):
        self.grid = np.asarray(grid, dtype=float)
        rate = 1.0 / float(np.median(np.diff(self.grid)))
        self.gain = float(gain)
        self.lead_samples = int(round(float(lead) * rate))
        self._b, self._a = butter(2, float(cutoff) / (rate / 2.0))

    def error(self, demo_q, t_measured, q_measured):
        """Demo minus measured angles on the grid; zero outside the measured span (a run with --until)."""
        measured = np.stack(
            [np.interp(self.grid, t_measured, q_measured[:, j]) for j in range(demo_q.shape[1])], axis=1
        )
        error = demo_q - measured
        error[(self.grid < t_measured[0]) | (self.grid > t_measured[-1])] = 0.0
        return error

    def update(self, offset, error):
        """The next run's offset from this run's offset and error."""
        shifted = np.roll(error, -self.lead_samples, axis=0)
        if self.lead_samples > 0:
            shifted[-self.lead_samples :] = error[-1]
        return offset + self.gain * filtfilt(self._b, self._a, shifted, axis=0)
