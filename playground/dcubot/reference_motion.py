"""Reference motion lookup interpolated between grid points (CHANGES #14).

Upstream's PolyReferenceMotion snaps the command to the nearest (dx, dy,
dtheta) grid point. DCUBOT's grid is 0.098 m/s apart in x and y, so a 0.15
m/s forward command imitated the 0.196 gait (0.206 m/s achieved) and a 0.09
strafe the 0.098 one (0.059 achieved). Here the polynomial coefficients of the
8 surrounding grid points are blended trilinearly; every gait in the pkl has
the same period, and a polynomial is linear in its coefficients, so this is
the same as blending the 8 reference trajectories frame by frame.
"""

import jax.numpy as jp

from playground.common.poly_reference_motion import PolyReferenceMotion


def _bracket(axis, v):
    """Lower grid index and blend weight of v on a sorted axis (clipped)."""
    axis = jp.array(axis)
    v = jp.clip(v, axis[0], axis[-1])
    i = jp.clip(jp.searchsorted(axis, v) - 1, 0, len(axis) - 2)
    w = (v - axis[i]) / (axis[i + 1] - axis[i])
    return i, w


class InterpolatedPolyReferenceMotion(PolyReferenceMotion):
    def get_reference_motion(self, dx, dy, dtheta, i):
        ix, wx = _bracket(self.dxs, dx)
        iy, wy = _bracket(self.dys, dy)
        it, wt = _bracket(self.dthetas, dtheta)
        coeffs = 0.0
        for ox, fx in ((0, 1 - wx), (1, wx)):
            for oy, fy in ((0, 1 - wy), (1, wy)):
                for ot, ft in ((0, 1 - wt), (1, wt)):
                    coeffs = coeffs + fx * fy * ft * self.data_array[ix + ox, iy + oy, it + ot]
        t = i % self.nb_steps_in_period / self.nb_steps_in_period
        return self.sample_polynomial(jp.clip(t, 0.0, 1.0), coeffs)
