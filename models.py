"""PSPL model with simple Earth-parallax terms.

Numba is optional; falls back to pure NumPy.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit as _njit
except Exception:  # pragma: no cover
    def _njit(func=None, **_):
        if func is None:
            return lambda f: f
        return func


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


class PSPL:
    """Point-source/point-lens magnification with optional parallax."""
    def __init__(self, ra_str, dec_str):
        # geometry set by source coordinates

        alpha_deg, delta_deg = hms_dms_to_deg(ra_str, dec_str)
        alpha_rad = alpha_deg * RADIAN_PER_DEGREE
        delta_rad = delta_deg * RADIAN_PER_DEGREE

        cos_alpha, sin_alpha = np.cos(alpha_rad), np.sin(alpha_rad)
        cos_delta, sin_delta = np.cos(delta_rad), np.sin(delta_rad)

        star_direction = np.array([cos_alpha * cos_delta, sin_alpha * cos_delta, sin_delta])
        north_pole = np.array([0.0, 0.0, 1.0])
        east_vec = np.cross(north_pole, star_direction)
        self.east = east_vec / np.linalg.norm(east_vec)
        self.north = np.cross(star_direction, self.east)

        spring = np.array([1.0, 0.0, 0.0])
        summer = np.array([0.0, 0.9174, 0.3971])
        phi_ref = (1.0 - PERIHELION_OFFSET_DAYS / 365.25) * TWO_PI
        psi_ref = _get_psi_njit(np.array(phi_ref), ECCENTRICITY)
        cos_ref = (np.cos(psi_ref) - ECCENTRICITY) / (1.0 - ECCENTRICITY * np.cos(psi_ref))
        sin_ref = -np.sqrt(1.0 - cos_ref**2)

        self.xpos = spring * cos_ref + summer * sin_ref
        self.ypos = -spring * sin_ref + summer * cos_ref

        self.qns_cache = []
        self.qes_cache = []
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

    def magnification(self, t, param, dataset_id=-1):
        """Compute PSPL magnification for times t and parameter dict."""
        t0 = param["t0"]
        u0 = param["u0"]
        tE = param.get("tE", abs(param["teff"] / u0))
        tau = (t - t0) / tE
        beta = u0
        pi1 = param.get("pi1", 0.0)
        pi2 = param.get("pi2", 0.0)
        pi_sq = pi1**2 + pi2**2
        if pi_sq > 0:
            qn, qe = self.get_parallax_components(t, t0, dataset_id)
            tau += qn * pi1 + qe * pi2
            beta += -qn * pi2 + qe * pi1
        u_sq = tau**2 + beta**2
        u_sq_p4 = u_sq + 4.0
        return (u_sq + 2.0) / np.sqrt(u_sq * u_sq_p4)
