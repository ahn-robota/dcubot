"""Left-right leg clearance for the leg_clearance cost (CHANGES #26).

MJX reports no distance for separated mesh pairs (margin/gap is not
implemented for meshes), so the clearance between the reduced leg hulls
(CHANGES #25, `<part>_hull`) is computed here: every hull vertex of one geom
against the face planes of the other, both ways, signed distance taken as
max over planes (exact inside and in face regions, about 1 mm off near
edges). Each hull has 32 vertices / 60 faces, so this is cheap.
"""

import jax.numpy as jp
import mujoco
import numpy as np


def _hull(m: mujoco.MjModel, geom: int):
    """Vertices and outward face planes (n, offset) of a mesh geom's convex hull, geom frame."""
    mid = m.geom_dataid[geom]
    g, adr = m.mesh_graph, m.mesh_graphadr[mid]
    nv, nf = g[adr], g[adr + 1]
    vert_globalid = g[adr + 2 + nv: adr + 2 + 2 * nv]
    face_globalid = g[adr + 2 + 3 * nv + 3 * nf: adr + 2 + 3 * nv + 6 * nf].reshape(nf, 3)
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    pts, tri = v[vert_globalid], v[face_globalid]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    off = (n * tri[:, 0]).sum(1)
    flip = np.sign((n * pts.mean(0)).sum(1) - off)  # centroid must be inside
    return pts, n * -flip[:, None], off * -flip


class LegClearance:
    """min signed distance (m) of each explicit `<pair>` in the model (the CHANGES #25 leg hulls)."""

    def __init__(self, m: mujoco.MjModel):
        self.pairs = [(int(a), int(b)) for a, b in zip(m.pair_geom1, m.pair_geom2)]
        hulls = {g: _hull(m, g) for p in self.pairs for g in p}
        self.hulls = {g: tuple(jp.asarray(x) for x in h) for g, h in hulls.items()}

    def _one_way(self, data, ga: int, gb: int):
        pa = self.hulls[ga][0]
        _, nb, ob = self.hulls[gb]
        world = data.geom_xpos[ga] + pa @ data.geom_xmat[ga].reshape(3, 3).T
        local = (world - data.geom_xpos[gb]) @ data.geom_xmat[gb].reshape(3, 3)
        return jp.min(jp.max(local @ nb.T - ob, axis=1))

    def __call__(self, data) -> jp.ndarray:
        return jp.stack(
            [jp.minimum(self._one_way(data, a, b), self._one_way(data, b, a)) for a, b in self.pairs]
        )
