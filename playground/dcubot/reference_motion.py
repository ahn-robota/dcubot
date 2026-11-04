"""Reference motion lookup by achieved speed, interpolated between grid points
(CHANGES #14, #15).

Upstream's PolyReferenceMotion snaps the command to the nearest (dx, dy,
dtheta) grid key. On DCUBOT's Placo sweep the keys are not the speeds the
gaits actually reach (strafe ~0.6x, turn 0.025 rad/s at key 0.98 and 0.11 at
1.961), and the grid is 0.098 m/s apart in x and y. Here:
  1. each command axis is mapped to the key whose gait reaches that speed,
     from the pure single-axis gaits' mean base velocity (clamped to the
     fastest gait on the grid);
  2. the polynomial coefficients of the 8 surrounding grid points are
     blended trilinearly. Every gait in the pkl has the same period, and a
     polynomial is linear in its coefficients, so this is the same as
     blending the 8 reference trajectories frame by frame.
"""

import jax.numpy as jp

from playground.common.poly_reference_motion import PolyReferenceMotion

# base_linear_vel x, y and base_angular_vel z in the reference frame.
_BASE_VEL_IDX = (28, 29, 33)


def _bracket(axis, v):
    """Lower grid index and blend weight of v on a sorted axis (clipped)."""
    axis = jp.array(axis)
    v = jp.clip(v, axis[0], axis[-1])
    i = jp.clip(jp.searchsorted(axis, v) - 1, 0, len(axis) - 2)
    w = (v - axis[i]) / (axis[i + 1] - axis[i])
    return i, w


class InterpolatedPolyReferenceMotion(PolyReferenceMotion):
    def __init__(self, polynomial_coefficients: str):
        super().__init__(polynomial_coefficients)
        ts = jp.arange(self.nb_steps_in_period) / self.nb_steps_in_period
        zero = [self.dxs.index(0.0), self.dys.index(0.0), self.dthetas.index(0.0)]
        self.achieved = []  # per axis: mean base velocity of each pure single-axis key
        for axis, keys in enumerate((self.dxs, self.dys, self.dthetas)):
            speeds = []
            for k in range(len(keys)):
                idx = list(zero)
                idx[axis] = k
                coeffs = self.data_array[idx[0], idx[1], idx[2]]
                frames = jp.stack([self.sample_polynomial(t, coeffs) for t in ts])
                speeds.append(float(frames[:, _BASE_VEL_IDX[axis]].mean()))
            self.achieved.append(jp.array(speeds))

    def speed_to_key(self, dx, dy, dtheta):
        keys = (self.dxs, self.dys, self.dthetas)
        return [
            jp.interp(v, self.achieved[a], jp.array(keys[a]))
            for a, v in enumerate((dx, dy, dtheta))
        ]

    def get_reference_motion(self, dx, dy, dtheta, i):
        kx, ky, kt = self.speed_to_key(dx, dy, dtheta)
        ix, wx = _bracket(self.dxs, kx)
        iy, wy = _bracket(self.dys, ky)
        it, wt = _bracket(self.dthetas, kt)
        coeffs = 0.0
        for ox, fx in ((0, 1 - wx), (1, wx)):
            for oy, fy in ((0, 1 - wy), (1, wy)):
                for ot, ft in ((0, 1 - wt), (1, wt)):
                    coeffs = coeffs + fx * fy * ft * self.data_array[ix + ox, iy + oy, it + ot]
        t = i % self.nb_steps_in_period / self.nb_steps_in_period
        return self.sample_polynomial(jp.clip(t, 0.0, 1.0), coeffs)
