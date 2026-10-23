"""System parameters for a monostatic OFDM ISAC radar (5G NR FR2-like numerology)."""
from dataclasses import dataclass, field

import numpy as np

C = 299_792_458.0  # speed of light [m/s]


@dataclass
class OFDMRadarConfig:
    fc: float = 28e9            # carrier frequency [Hz]
    scs: float = 120e3          # subcarrier spacing [Hz] (NR numerology mu=3)
    n_sc: int = 512             # number of subcarriers used for sensing
    n_sym: int = 128            # number of OFDM symbols in the frame (slow time)
    n_rx: int = 8               # receive ULA elements
    d_over_lambda: float = 0.5  # ULA spacing in wavelengths
    cp_ratio: float = 0.07      # NR normal CP is ~7% of the useful symbol
    modulation: str = "qpsk"
    seed: int = 0

    # FFT sizes (zero padding for smoother maps)
    n_range_fft: int = 2048
    n_doppler_fft: int = 512
    n_angle_fft: int = 64

    @property
    def pad_factors(self):
        """Zero-padding factors (range, Doppler) used to scale CFAR windows."""
        return self.n_range_fft // self.n_sc, self.n_doppler_fft // self.n_sym

    @property
    def wavelength(self) -> float:
        return C / self.fc

    @property
    def bandwidth(self) -> float:
        return self.n_sc * self.scs

    @property
    def t_sym(self) -> float:
        """Total OFDM symbol duration including CP [s]."""
        return (1.0 + self.cp_ratio) / self.scs

    @property
    def range_resolution(self) -> float:
        return C / (2 * self.bandwidth)

    @property
    def max_range(self) -> float:
        """Max unambiguous range from subcarrier spacing [m]."""
        return C / (2 * self.scs)

    @property
    def cp_limited_range(self) -> float:
        """Range beyond which echoes exceed the CP (ISI) [m]."""
        return C * self.cp_ratio / self.scs / 2

    @property
    def velocity_resolution(self) -> float:
        return self.wavelength / (2 * self.n_sym * self.t_sym)

    @property
    def max_velocity(self) -> float:
        return self.wavelength / (4 * self.t_sym)

    def range_axis(self) -> np.ndarray:
        return np.arange(self.n_range_fft) * C / (2 * self.scs * self.n_range_fft)

    def velocity_axis(self) -> np.ndarray:
        fd = np.fft.fftshift(np.fft.fftfreq(self.n_doppler_fft, d=self.t_sym))
        return fd * self.wavelength / 2

    def angle_axis_deg(self) -> np.ndarray:
        u = np.fft.fftshift(np.fft.fftfreq(self.n_angle_fft, d=self.d_over_lambda))
        return np.degrees(np.arcsin(np.clip(u, -1, 1)))

    def summary(self) -> str:
        return (
            f"fc={self.fc/1e9:.1f} GHz, B={self.bandwidth/1e6:.1f} MHz, "
            f"N_sc={self.n_sc}, M_sym={self.n_sym}, N_rx={self.n_rx}\n"
            f"range res={self.range_resolution:.2f} m, CP-limited range={self.cp_limited_range:.1f} m\n"
            f"velocity res={self.velocity_resolution:.2f} m/s, max |v|={self.max_velocity:.1f} m/s"
        )


@dataclass
class Target:
    range_m: float
    velocity_mps: float
    angle_deg: float
    rcs_db: float = 0.0  # relative reflectivity [dB]


@dataclass
class Scene:
    targets: list = field(default_factory=list)
    # static clutter: (range_m, angle_deg, power_db) with zero Doppler
    clutter: list = field(default_factory=list)
    snr_db: float = 10.0  # per-element SNR of a 0 dB target
