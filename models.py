"""Microlensing models and utilities.

- Parallax: computes projected Earth parallax components for a given source.
- PSPL: point-source point-lens magnification with optional parallax.
- PSBL: point-source binary-lens magnification via VBBinaryLensing.BinaryMag2.

Numba is optional; falls back to pure NumPy.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit as _njit
except Exception:  # pragma: no cover

    def _njit(func=None, **_):
        if func is None:
            return lambda f: f
        return func


from . import VBBinaryLensing as _VBB

vbbl = _VBB.VBBinaryLensing()


ECCENTRICITY = 0.0167
SQRT_1_MINUS_ECC_SQ = np.sqrt(1.0 - ECCENTRICITY**2)
VERNAL_EQUINOX_JD = 2719.55
PERIHELION_OFFSET_DAYS = 75
PERIHELION_JD = VERNAL_EQUINOX_JD - PERIHELION_OFFSET_DAYS
RADIAN_PER_DEGREE = np.pi / 180.0
TWO_PI = 2.0 * np.pi


@_njit
def _get_psi_njit(phi, ecc):
    psi = (phi + np.pi) % TWO_PI - np.pi
    for _ in range(4):
        fun = psi - ecc * np.sin(psi)
        diff = phi - fun
        psi += diff / (1.0 - ecc * np.cos(psi))
    return psi


@_njit
def _get_projected_sun_pos_njit(t, peri_jd, p, ecc, xpos, ypos, sqrt_term):
    phi = (t - peri_jd) / 365.25 * p
    psi = _get_psi_njit(phi, ecc)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    if t.ndim == 0:
        sun = xpos * (cos_psi - ecc) + ypos * sin_psi * sqrt_term
    else:
        sun = xpos[:, None] * (cos_psi - ecc) + ypos[:, None] * sin_psi * sqrt_term
    return sun


def hms_dms_to_deg(ra_str, dec_str):
    ra_h, ra_m, ra_s = [float(x) for x in ra_str.split(":")]
    ra_deg = (ra_h + ra_m / 60.0 + ra_s / 3600.0) * 15.0
    sign = -1.0 if "-" in dec_str else 1.0
    dec_d, dec_m, dec_s = [float(x) for x in dec_str.replace("-", "").split(":")]
    dec_deg = sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0)
    return ra_deg, dec_deg


class Parallax:
    """Compute projected parallax components for a source direction.

    Builds local east/north basis from the source coordinates and provides
    time-dependent projections (q_n, q_e) of the Earth's orbital motion
    relative to a chosen reference epoch.
    """

    def __init__(self, ra_str: str, dec_str: str):
        # geometry set by source coordinates
        alpha_deg, delta_deg = hms_dms_to_deg(ra_str, dec_str)
        alpha_rad = alpha_deg * RADIAN_PER_DEGREE
        delta_rad = delta_deg * RADIAN_PER_DEGREE

        cos_alpha, sin_alpha = np.cos(alpha_rad), np.sin(alpha_rad)
        cos_delta, sin_delta = np.cos(delta_rad), np.sin(delta_rad)

        star_direction = np.array(
            [cos_alpha * cos_delta, sin_alpha * cos_delta, sin_delta]
        )
        north_pole = np.array([0.0, 0.0, 1.0])
        east_vec = np.cross(north_pole, star_direction)
        self.east = east_vec / np.linalg.norm(east_vec)
        self.north = np.cross(star_direction, self.east)

        spring = np.array([1.0, 0.0, 0.0])
        summer = np.array([0.0, 0.9174, 0.3971])
        phi_ref = (1.0 - PERIHELION_OFFSET_DAYS / 365.25) * TWO_PI
        psi_ref = _get_psi_njit(np.array(phi_ref), ECCENTRICITY)
        cos_ref = (np.cos(psi_ref) - ECCENTRICITY) / (
            1.0 - ECCENTRICITY * np.cos(psi_ref)
        )
        sin_ref = -np.sqrt(1.0 - cos_ref**2)

        self.xpos = spring * cos_ref + summer * sin_ref
        self.ypos = -spring * sin_ref + summer * cos_ref

        self.qns_cache: list[np.ndarray] = []
        self.qes_cache: list[np.ndarray] = []
        self.is_precal = False

    def precalculate_parallax(self, t_list):
        """Cache projected Sun positions for each dataset time array."""
        self.qns_cache.clear()
        self.qes_cache.clear()
        for t_arr in t_list:
            sun_pos = _get_projected_sun_pos_njit(
                t_arr,
                PERIHELION_JD,
                TWO_PI,
                ECCENTRICITY,
                self.xpos,
                self.ypos,
                SQRT_1_MINUS_ECC_SQ,
            )
            self.qns_cache.append(self.north @ sun_pos)
            self.qes_cache.append(self.east @ sun_pos)
        self.is_precal = True

    def get_parallax_components(self, t, t0, dataset_id):
        """Return (qn, qe) projected components relative to t0."""
        sun_pos_t0_plus_1 = _get_projected_sun_pos_njit(
            np.array(t0 + 1.0),
            PERIHELION_JD,
            TWO_PI,
            ECCENTRICITY,
            self.xpos,
            self.ypos,
            SQRT_1_MINUS_ECC_SQ,
        )
        sun_pos_t0_minus_1 = _get_projected_sun_pos_njit(
            np.array(t0 - 1.0),
            PERIHELION_JD,
            TWO_PI,
            ECCENTRICITY,
            self.xpos,
            self.ypos,
            SQRT_1_MINUS_ECC_SQ,
        )
        sun_pos_t0 = _get_projected_sun_pos_njit(
            np.array(t0),
            PERIHELION_JD,
            TWO_PI,
            ECCENTRICITY,
            self.xpos,
            self.ypos,
            SQRT_1_MINUS_ECC_SQ,
        )

        qn2 = self.north @ sun_pos_t0_plus_1
        qe2 = self.east @ sun_pos_t0_plus_1
        qn1 = self.north @ sun_pos_t0_minus_1
        qe1 = self.east @ sun_pos_t0_minus_1
        qn0 = self.north @ sun_pos_t0
        qe0 = self.east @ sun_pos_t0

        if dataset_id >= 0 and self.is_precal:
            qn, qe = self.qns_cache[dataset_id], self.qes_cache[dataset_id]
        else:
            sun_pos = _get_projected_sun_pos_njit(
                t,
                PERIHELION_JD,
                TWO_PI,
                ECCENTRICITY,
                self.xpos,
                self.ypos,
                SQRT_1_MINUS_ECC_SQ,
            )
            qn, qe = self.north @ sun_pos, self.east @ sun_pos

        dt = t - t0
        qn_final = qn - (qn0 + (qn2 - qn1) * dt * 0.5)
        qe_final = qe - (qe0 + (qe2 - qe1) * dt * 0.5)
        return qn_final, qe_final


class SingleLens(Parallax):
    """Point-source/point-lens magnification with optional parallax.

    Inherits geometry and parallax trajectory from Parallax; adds the PSPL
    magnification calculation.
    """

    @staticmethod
    def normalize(param: Dict) -> Dict:
        param = param.copy()
        if "tE" not in param:
            param["tE"] = abs(param["teff"] / param["u0"])
        param["pi1"] = param.get("pi1", 0.0)
        param["pi2"] = param.get("pi2", 0.0)

        return param

    @staticmethod
    def north_east(param: Dict) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # Direction of piE source -> lens
        piEN, piEE = param["pi1"], param["pi2"]  # parallax components in EN frame
        piE = np.hypot(piEN, piEE)
        if piE == 0.0:
            return (np.nan, np.nan), (np.nan, np.nan)

        # PA of the \mu_LS
        # Phi_pi = np.arctan2(piEE, piEN)

        # partial (tau, u) / partial qn -> North
        tau_north, u_north = piEN / piE, -piEE / piE

        # partial (tau, u) / partial qe -> East
        tau_east, u_east = piEE / piE, piEN / piE
        return (tau_north, u_north), (tau_east, u_east)

    def trajectory(
        self, t: NDArray, param: Dict, dataset_id: int = -1
    ) -> Tuple[NDArray, NDArray]:
        param = self.normalize(param)
        t0, u0, tE = param["t0"], param["u0"], param["tE"]
        piEN, piEE = param["pi1"], param["pi2"]  # parallax components in EN frame

        tau = (t - t0) / tE
        u = u0

        if (piEN * piEN + piEE * piEE) > 0.0:
            qn, qe = self.get_parallax_components(t, t0, dataset_id)
            tau += qn * piEN + qe * piEE
            u += -qn * piEE + qe * piEN

        return tau, u

    def magnification(self, t: NDArray, param: Dict, dataset_id: int = -1) -> NDArray:
        """Compute PSPL magnification for times t and parameter dict.

        Expected keys in param: t0, u0, and either tE or (teff,u0).
        Optional parallax vector components: pi1, pi2.
        """

        tau, u = self.trajectory(t, param, dataset_id)

        u_sq = tau * tau + u * u
        u_sq_p4 = u_sq + 4.0

        return (u_sq + 2.0) / np.sqrt(u_sq * u_sq_p4)

    def images(self, t: NDArray, param: Dict, thetaE: float = 1, dataset_id: int = -1):
        """Compute PSPL major-to-minor vectors (x, y, eta) in u, v plane for times t and parameter dict."""
        taup, up = self.trajectory(t, param, dataset_id)
        u_sq = taup * taup + up * up
        u_sq_p4 = u_sq + 4.0
        A = (u_sq + 2.0) / np.sqrt(u_sq * u_sq_p4)
        eta = (A - 1) / (A + 1)
        sep = eta**0.25 + eta**-0.25  # image separation in thetaE units

        (tau_north, u_north), (tau_east, u_east) = self.north_east(param)

        # tau, u -> north, east
        det = tau_east * u_north - tau_north * u_east
        ue = (u_north * taup - tau_north * up) / det  # east
        un = (-u_east * taup + tau_east * up) / det  # north

        # source position -> major-to-minor vector
        dn, de = -sep * un / u_sq**0.5, -sep * ue / u_sq**0.5
        return dn * thetaE, de * thetaE, eta


class BinaryLens(Parallax):
    """Point-source binary-lens magnification using VBBinaryLensing.

    - Provides the same interface as PSPL: `.precalculate_parallax` and
      `.magnification(t, params, dataset_id=...)`.
    - Computes (tau, beta) with optional parallax (pi1, pi2) using Parallax.
    - Rotates trajectory by `alpha` (radians) to get (x, y) in lens frame.
    - Calls `VBBinaryLensing.BinaryMag2` on (s, q, x, y).

    Expected params keys:
      t0, u0, tE, s, q, alpha[, rho, pi1, pi2]
    """

    def __init__(
        self, ra_str: str, dec_str: str, rel_tol: float = 1e-3, tol: float = 1e-3
    ):
        # Initialize parallax geometry
        super().__init__(ra_str, dec_str)
        vbbl.Tol = tol
        vbbl.RelTol = rel_tol

    @staticmethod
    def normalize(param: Dict) -> Dict:
        param = param.copy()
        param["rho"] = param.get("rho", 10 ** param.get("logrho", -6))
        param["q"] = param.get("q", 10 ** param.get("logq", -4))
        param["s"] = param.get("s", 10 ** param.get("logs", -4))
        if "tE" not in param:
            param["tE"] = abs(param["teff"] / param["u0"])
        return param

    def trajectory(
        self, t: NDArray, param: Dict, dataset_id: int = -1
    ) -> Tuple[NDArray, NDArray]:
        """Source Position (y1, y2) relative to Binary-Lens c.o.m."""

        t0, u0, tE = param["t0"], param["u0"], param["tE"]
        alpha = param["alpha"]  # radians
        piEN, piEE = param["pi1"], param["pi2"]  # parallax components in EN frame

        tau = (t - t0) / tE
        u = u0

        if (piEN * piEN + piEE * piEE) > 0.0:
            qn, qe = self.get_parallax_components(t, t0, dataset_id)
            tau += +qn * piEN + qe * piEE
            u += +qe * piEN - qn * piEE

        calpha, salpha = np.cos(alpha), np.sin(alpha)
        y1 = +u * salpha - tau * calpha
        y2 = -u * calpha - tau * salpha

        return y1, y2

    @staticmethod
    def north_east(param: Dict) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # Direction of piE source -> lens
        piEN, piEE = param["pi1"], param["pi2"]  # parallax components in EN frame
        alpha = param["alpha"]  # radians
        calpha, salpha = np.cos(alpha), np.sin(alpha)

        # PA of the \mu_LS
        # Phi_pi = np.arctan2(piEE, piEN)

        # partial (tau, u) / partial qn -> North
        tau, u = piEN, -piEE
        y1_north = +u * salpha - tau * calpha
        y2_north = -u * calpha - tau * salpha
        y1_north, y2_north = -y1_north, -y2_north  # flip as y1, y2 are source position

        # partial (tau, u) / partial qe -> East
        tau, u = piEE, piEN
        y1_east = +u * salpha - tau * calpha
        y2_east = -u * calpha - tau * salpha
        y1_east, y2_east = -y1_east, -y2_east  # flip as y1, y2 are source position

        return (y1_north, y2_north), (y1_east, y2_east)

    def magnification(self, t: NDArray, param: Dict, dataset_id: int = -1) -> NDArray:
        param = self.normalize(param)
        s, q = param["s"], param["q"]
        s_t = np.ones_like(t) * s
        rho = param["rho"]
        y1, y2 = self.trajectory(t, param, dataset_id)

        return np.array(vbbl.BinaryMag2_vec(s_t, q, y1, y2, rho))


class BinaryLensOrb(BinaryLens):

    @staticmethod
    def beta(param: dict, piS: float = 0.125, thetaE: float = 1) -> float:
        kappa = 8.144
        piE = np.hypot(param["pi1"], param["pi2"])
        gamma2 = (param["ds_dt"] / param["s"]) ** 2 + (param["dalpha_dt"]) ** 2
        beta = kappa * (365.25**2) / 8 / np.pi**2
        beta *= piE / thetaE * gamma2 * (param["s"] / (piE + piS / thetaE)) ** 3
        return beta

    def magnification(self, t: NDArray, param: Dict, dataset_id: int = -1) -> NDArray:
        param = self.normalize(param)
        s, q = param["s"], param["q"]
        ds_dt, dalpha_dt = param["ds_dt"], param["dalpha_dt"]
        rho = param["rho"]

        dt = t - param["t_kep"]
        dt[(t < param["t_min"]) | (t > param["t_max"])] = 0.0

        s_t = s + ds_dt * dt
        alpha_t = param["alpha"] + dalpha_dt * dt
        param_rot = param.copy()
        param_rot["alpha"] = alpha_t

        y1, y2 = self.trajectory(t, param_rot, dataset_id)
        return np.array(vbbl.BinaryMag2_vec(s_t, q, y1, y2, rho))


if __name__ == "__main__":
    coord = "17:31:42.61 -30:46:17.04"
    model = SingleLens(*coord.split())
    t = np.array([10096])
    param = {
        "t0": 10097.7070,
        "u0": 0.68033,
        "teff": 48.4662,
        "pi1": 0.0799,
        "pi2": -0.0632,
    }
    print(model.images(t, param))
