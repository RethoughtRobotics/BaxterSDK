"""How far each joint trails its raw position command: a linear model of the command's recent velocity.

  Ljung, "System Identification: Theory for the User", 2nd ed., Prentice Hall 1999 (FIR models, ch. 4).
  Nguyen-Tuong, Peters, "Model learning for robot control: a survey", Cognitive Processing 2011.

In RAW_POSITION_MODE with the inverse dynamics feedforward, Baxter's joints follow the commanded
angle closely but not exactly: friction, the series elastic springs and the joint servo leave each
joint behind by an amount that grows with how the command has been moving. This module models that
tracking error, per joint, on a uniform time grid of period dt:

    e[n] = q_cmd[n] - q[n] = c + sum_{k=0}^{taps-1} h[k] * v[n - k]

v is the commanded velocity, h the joint's finite impulse response (FIR) and c a constant. A FIR
assumes nothing about the error's form: any linear, time-invariant lag (a delay, a first-order lag,
a spring that rings) is a weighted sum of recent velocities, and the weights are fitted from data
by least squares. With taps=1 the model reduces to e = c + h * v, a velocity-proportional lag.

The joints are modelled independently. Whether that is enough (or the error also depends on the
other joints and the pose) is answered by `explained` on motions the model was not fitted to.
"""

import numpy as np

JOINTS = ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')


def lagged(v, taps):
    """Rows [v[n], v[n-1], ..., v[n-taps+1]] for n = taps-1 .. len(v)-1: the regressors of a FIR."""
    v = np.asarray(v, dtype=float)
    return np.stack([v[taps - 1 - k : len(v) - k] for k in range(taps)], axis=1)


class JointErrorModel:
    """One FIR per joint from commanded velocity to tracking error, fitted on segments of a uniform grid.

    A segment is a stretch of continuous raw streaming, (v, e), each (N, joints) on the grid. Rows
    are built inside each segment only, so a FIR never reaches across a gap in the data.
    """

    def __init__(self, taps=30, dt=0.01, ridge=1e-3):
        self.taps, self.dt, self.ridge = int(taps), float(dt), float(ridge)
        self.h = None  # (joints, taps): the impulse responses, in s (error in rad per rad/s of velocity)
        self.c = None  # (joints,): constant error, rad

    def _rows(self, segments, joint):
        X, y = [], []
        for v, e in segments:
            if len(v) < self.taps:
                continue
            X.append(lagged(v[:, joint], self.taps))
            y.append(e[self.taps - 1 :, joint])
        return np.vstack(X), np.concatenate(y)

    def fit(self, segments):
        """Least squares with a small ridge penalty on h (not on c), per joint."""
        joints = segments[0][0].shape[1]
        self.h, self.c = np.zeros((joints, self.taps)), np.zeros(joints)
        for j in range(joints):
            X, y = self._rows(segments, j)
            X = np.c_[X, np.ones(len(X))]
            penalty = self.ridge * len(X) * np.eye(self.taps + 1)
            penalty[-1, -1] = 0.0
            coef = np.linalg.solve(X.T @ X + penalty, X.T @ y)
            self.h[j], self.c[j] = coef[:-1], coef[-1]
        return self

    def predict(self, v):
        """Predicted error (N, joints) for a commanded velocity (N, joints); NaN for the first taps-1 rows."""
        v = np.asarray(v, dtype=float)
        e = np.full(v.shape, np.nan)
        for j in range(v.shape[1]):
            e[self.taps - 1 :, j] = lagged(v[:, j], self.taps) @ self.h[j] + self.c[j]
        return e

    def explained(self, segments):
        """Per joint: (rms error, rms error left after subtracting the prediction, fraction explained),
        over every modelled row of the segments. Rad, rad, unitless. The fraction is of the mean square
        error, offset included (1 - left^2 / rms^2): a constant the model gets wrong counts against it."""
        out = []
        for j in range(len(self.c)):
            X, y = self._rows(segments, j)
            residual = y - (X @ self.h[j] + self.c[j])
            rms, left = np.sqrt(np.mean(y**2)), np.sqrt(np.mean(residual**2))
            out.append((rms, left, 1.0 - left**2 / rms**2))
        return out

    def lag(self):
        """Per joint, the steady lag behind a constant velocity, sum(h), in s: the FIR's DC gain."""
        return self.h.sum(axis=1)

    def save(self, path):
        np.savez(path, h=self.h, c=self.c, taps=self.taps, dt=self.dt, ridge=self.ridge)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        model = cls(int(d['taps']), float(d['dt']), float(d['ridge']))
        model.h, model.c = d['h'], d['c']
        return model
