"""Frequency-domain echo model for a monostatic OFDM radar with a ULA receiver.

After CP removal and FFT, the echo on subcarrier n, symbol m, antenna k is

    Y[n, m, k] = X[n, m] * sum_t a_t * exp(-j2pi n df tau_t)
                                     * exp( j2pi f_D,t m T_sym)
                                     * exp( j2pi k d sin(theta_t) / lambda) + W[n, m, k]

valid while every echo delay stays inside the cyclic prefix.
"""
import numpy as np

from .config import C, OFDMRadarConfig, Scene


def random_symbols(cfg: OFDMRadarConfig, rng: np.random.Generator) -> np.ndarray:
    """Random unit-power data symbols X with shape (n_sc, n_sym)."""
    shape = (cfg.n_sc, cfg.n_sym)
    if cfg.modulation == "qpsk":
        bits = rng.integers(0, 2, size=shape + (2,))
        return ((2 * bits[..., 0] - 1) + 1j * (2 * bits[..., 1] - 1)) / np.sqrt(2)
    if cfg.modulation == "16qam":
        lv = np.array([-3, -1, 1, 3])
        return (rng.choice(lv, shape) + 1j * rng.choice(lv, shape)) / np.sqrt(10)
    raise ValueError(f"unsupported modulation {cfg.modulation}")


def _steering(cfg, range_m, velocity_mps, angle_deg):
    n = np.arange(cfg.n_sc)[:, None, None]
    m = np.arange(cfg.n_sym)[None, :, None]
    k = np.arange(cfg.n_rx)[None, None, :]
    tau = 2 * range_m / C
    fd = 2 * velocity_mps / cfg.wavelength
    u = np.sin(np.radians(angle_deg))
    return (
        np.exp(-1j * 2 * np.pi * n * cfg.scs * tau)
        * np.exp(1j * 2 * np.pi * fd * m * cfg.t_sym)
        * np.exp(1j * 2 * np.pi * cfg.d_over_lambda * k * u)
    )


def simulate_echo(cfg: OFDMRadarConfig, scene: Scene, rng=None):
    """Return (X, Y): transmitted symbols (n_sc, n_sym), received grid (n_sc, n_sym, n_rx)."""
    rng = np.random.default_rng(cfg.seed) if rng is None else rng
    X = random_symbols(cfg, rng)
    H = np.zeros((cfg.n_sc, cfg.n_sym, cfg.n_rx), dtype=complex)

    noise_power = 10 ** (-scene.snr_db / 10)
    for t in scene.targets:
        amp = 10 ** (t.rcs_db / 20) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        H += amp * _steering(cfg, t.range_m, t.velocity_mps, t.angle_deg)
    for r, ang, p_db in scene.clutter:
        amp = 10 ** (p_db / 20) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        H += amp * _steering(cfg, r, 0.0, ang)

    W = np.sqrt(noise_power / 2) * (
        rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape)
    )
    Y = X[..., None] * H + W
    return X, Y
