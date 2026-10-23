"""Classical OFDM radar processing chain.

1. radar cube     : element-wise division Y / X removes the data (reciprocal filter)
2. clutter removal: MTI / static clutter removal along slow time
3. range          : IFFT over subcarriers
4. Doppler        : FFT over OFDM symbols
5. CFAR           : 2D cell-averaging CFAR on the range-Doppler map
6. angle          : FFT beamforming or MUSIC across the ULA, per detection
"""
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

from .config import OFDMRadarConfig


# 1. Radar cube ---------------------------------------------------------------
def radar_cube(Y: np.ndarray, X: np.ndarray, method: str = "zf", eps: float = 1e-12):
    """Remove the transmitted data from the received grid.

    zf  : reciprocal filter Y / X (classic OFDM radar, flat noise floor for PSK)
    mf  : matched filter Y * conj(X) (better for QAM with varying amplitude)
    """
    if method == "zf":
        return Y / (X[..., None] + eps)
    if method == "mf":
        return Y * np.conj(X)[..., None]
    raise ValueError(method)


# 2. Clutter suppression ------------------------------------------------------
def static_clutter_removal(cube: np.ndarray) -> np.ndarray:
    """Subtract the slow-time mean, i.e. remove the zero-Doppler component."""
    return cube - cube.mean(axis=1, keepdims=True)


def mti_filter(cube: np.ndarray, order: int = 2) -> np.ndarray:
    """Binomial pulse canceller along slow time (order=1 → 2-pulse, order=2 → 3-pulse)."""
    out = cube
    for _ in range(order):
        out = np.diff(out, axis=1)
    return out


# 3-4. Range / Doppler --------------------------------------------------------
def range_doppler(cube: np.ndarray, cfg: OFDMRadarConfig, window: bool = True):
    """Return the complex range-Doppler-antenna tensor (n_range_fft, n_doppler_fft, n_rx)."""
    n_sc, n_sym, _ = cube.shape
    x = cube
    if window:
        x = x * np.hanning(n_sc)[:, None, None] * np.hanning(n_sym)[None, :, None]
    rng_profile = np.fft.ifft(x, n=cfg.n_range_fft, axis=0)
    rd = np.fft.fftshift(np.fft.fft(rng_profile, n=cfg.n_doppler_fft, axis=1), axes=1)
    return rd


def rd_power_db(rd: np.ndarray) -> np.ndarray:
    """Non-coherent integration over antennas, in dB."""
    p = np.mean(np.abs(rd) ** 2, axis=2)
    return 10 * np.log10(p + 1e-20)


# 5. CFAR ---------------------------------------------------------------------
def ca_cfar_2d(power_lin: np.ndarray, guard=(4, 2), train=(8, 4), pfa: float = 1e-5):
    """2D cell-averaging CFAR. Returns (detection mask, threshold).

    guard/train are half-widths along (range, Doppler). Doppler axis wraps.
    """
    g_r, g_d = guard
    t_r, t_d = train
    outer = (2 * (g_r + t_r) + 1, 2 * (g_d + t_d) + 1)
    inner = (2 * g_r + 1, 2 * g_d + 1)
    n_outer = outer[0] * outer[1]
    n_inner = inner[0] * inner[1]
    n_train = n_outer - n_inner

    sum_outer = uniform_filter(power_lin, size=outer, mode=("nearest", "wrap")) * n_outer
    sum_inner = uniform_filter(power_lin, size=inner, mode=("nearest", "wrap")) * n_inner
    noise = (sum_outer - sum_inner) / n_train

    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    thr = alpha * noise
    return power_lin > thr, thr


def cluster_peaks(power_lin: np.ndarray, mask: np.ndarray, size=(5, 3)):
    """Keep only local maxima inside the CFAR mask (one detection per target)."""
    local_max = power_lin == maximum_filter(power_lin, size=size, mode=("nearest", "wrap"))
    idx = np.argwhere(mask & local_max)
    order = np.argsort(-power_lin[idx[:, 0], idx[:, 1]])
    return idx[order]


# 6. Angle estimation -----------------------------------------------------------
def angle_fft(snapshot: np.ndarray, cfg: OFDMRadarConfig) -> np.ndarray:
    """Angle spectrum (dB) of one antenna snapshot via zero-padded FFT."""
    spec = np.fft.fftshift(np.fft.fft(snapshot * np.hanning(cfg.n_rx), n=cfg.n_angle_fft))
    return 20 * np.log10(np.abs(spec) + 1e-20)


def angle_music(rd: np.ndarray, r_idx: int, d_idx: int, cfg: OFDMRadarConfig,
                n_src: int = 1, grid_deg=np.linspace(-80, 80, 641), nbhd=(2, 2)):
    """MUSIC pseudo-spectrum using neighbouring RD cells as snapshots."""
    r0, r1 = max(r_idx - nbhd[0], 0), r_idx + nbhd[0] + 1
    d = np.arange(d_idx - nbhd[1], d_idx + nbhd[1] + 1) % rd.shape[1]
    snaps = rd[r0:r1][:, d].reshape(-1, cfg.n_rx).T  # (n_rx, n_snap)
    R = snaps @ snaps.conj().T / snaps.shape[1]
    _, vecs = np.linalg.eigh(R)
    En = vecs[:, : cfg.n_rx - n_src]
    k = np.arange(cfg.n_rx)[:, None]
    A = np.exp(1j * 2 * np.pi * cfg.d_over_lambda * k * np.sin(np.radians(grid_deg))[None, :])
    p = 1.0 / np.sum(np.abs(En.conj().T @ A) ** 2, axis=0)
    return grid_deg, 10 * np.log10(p / p.max())


# Full chain ------------------------------------------------------------------
def detect_targets(Y, X, cfg: OFDMRadarConfig, clutter: str = "static",
                   pfa: float = 1e-5, max_targets: int = 20, angle_method: str = "fft"):
    """Run the whole chain and return a list of detections plus intermediate maps."""
    cube = radar_cube(Y, X)
    if clutter == "static":
        cube = static_clutter_removal(cube)
    elif clutter == "mti":
        cube = mti_filter(cube)
    rd = range_doppler(cube, cfg)
    p_lin = np.mean(np.abs(rd) ** 2, axis=2)
    pr, pd = cfg.pad_factors  # CFAR windows defined in resolution cells, scaled to FFT bins
    mask, thr = ca_cfar_2d(p_lin, guard=(3 * pr, 2 * pd), train=(6 * pr, 4 * pd), pfa=pfa)
    peaks = cluster_peaks(p_lin, mask, size=(4 * pr + 1, 3 * pd + 1))[:max_targets]

    r_axis, v_axis, a_axis = cfg.range_axis(), cfg.velocity_axis(), cfg.angle_axis_deg()
    dets = []
    for r_i, d_i in peaks:
        if angle_method == "music":
            grid, spec = angle_music(rd, r_i, d_i, cfg)
            ang = grid[np.argmax(spec)]
        else:
            ang = a_axis[np.argmax(angle_fft(rd[r_i, d_i], cfg))]
        dets.append({
            "range_m": float(r_axis[r_i]),
            "velocity_mps": float(v_axis[d_i]),
            "angle_deg": float(ang),
            "power_db": float(10 * np.log10(p_lin[r_i, d_i])),
            "snr_over_thr_db": float(10 * np.log10(p_lin[r_i, d_i] / thr[r_i, d_i])),
        })
    return dets, {"rd": rd, "power": p_lin, "mask": mask, "threshold": thr}
