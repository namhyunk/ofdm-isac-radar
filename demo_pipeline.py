"""End-to-end demo of the classical OFDM radar chain.

    python demo_pipeline.py            # saves figures to figures/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ofdm_radar import OFDMRadarConfig, Scene, Target, detect_targets, simulate_echo
from ofdm_radar.processing import (angle_fft, angle_music, radar_cube, range_doppler,
                                   rd_power_db)

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

cfg = OFDMRadarConfig()
print(cfg.summary(), "\n")

scene = Scene(
    targets=[
        Target(range_m=18.0, velocity_mps=12.0, angle_deg=-20.0, rcs_db=0.0),
        Target(range_m=35.5, velocity_mps=-8.0, angle_deg=15.0, rcs_db=-3.0),
        Target(range_m=52.0, velocity_mps=25.0, angle_deg=40.0, rcs_db=-6.0),
        Target(range_m=35.5, velocity_mps=4.0, angle_deg=-45.0, rcs_db=-3.0),  # same range as #2
    ],
    clutter=[(10.0, 0.0, 25.0), (28.0, -30.0, 22.0), (45.0, 55.0, 20.0)],  # strong static returns
    snr_db=-15.0,
)

X, Y = simulate_echo(cfg, scene)

# Run the chain with and without clutter suppression
dets, maps = detect_targets(Y, X, cfg, clutter="static", angle_method="music")
dets_raw, maps_raw = detect_targets(Y, X, cfg, clutter="none")

print("Ground truth:")
for t in scene.targets:
    print(f"  R={t.range_m:6.2f} m  v={t.velocity_mps:6.2f} m/s  θ={t.angle_deg:6.1f}°")
print(f"\nDetections with static clutter removal ({len(dets)}):")
for d in dets:
    print(f"  R={d['range_m']:6.2f} m  v={d['velocity_mps']:6.2f} m/s  θ={d['angle_deg']:6.1f}°"
          f"  ({d['snr_over_thr_db']:.1f} dB over CFAR thr)")
print(f"\nDetections WITHOUT clutter removal: {len(dets_raw)} "
      f"(zero-Doppler ones: {sum(abs(d['velocity_mps']) < cfg.velocity_resolution for d in dets_raw)})")

# ---------------------------------------------------------------- figures
r_ax, v_ax = cfg.range_axis(), cfg.velocity_axis()
r_lim = cfg.cp_limited_range
ext = [v_ax[0], v_ax[-1], r_ax[0], r_ax[-1]]

fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
for ax, m, title in [
    (axs[0], maps_raw["power"], "RD map, no clutter removal"),
    (axs[1], maps["power"], "RD map, static clutter removal"),
]:
    im = ax.imshow(10 * np.log10(m), aspect="auto", origin="lower", extent=ext, cmap="viridis",
                   vmax=10 * np.log10(m.max()), vmin=10 * np.log10(m.max()) - 50)
    ax.set_ylim(0, r_lim); ax.set_xlim(-60, 60)
    ax.set(title=title, xlabel="velocity [m/s]", ylabel="range [m]")
    fig.colorbar(im, ax=ax, label="dB")
axs[2].imshow(maps["mask"], aspect="auto", origin="lower", extent=ext, cmap="gray_r")
axs[2].set_ylim(0, r_lim); axs[2].set_xlim(-60, 60)
for t in scene.targets:
    axs[2].plot(t.velocity_mps, t.range_m, "o", mfc="none", mec="tab:red", ms=14, mew=2)
for d in dets:
    axs[2].plot(d["velocity_mps"], d["range_m"], "x", c="tab:blue", ms=10, mew=2)
axs[2].set(title="CA-CFAR mask (o truth, x detection)", xlabel="velocity [m/s]", ylabel="range [m]")
fig.savefig(OUT / "range_doppler_cfar.png", dpi=130)

# Angle spectra for the detection at 35.5 m (two targets share this range bin)
rd = maps["rd"]
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for d in dets[:4]:
    r_i = int(np.argmin(np.abs(r_ax - d["range_m"])))
    d_i = int(np.argmin(np.abs(v_ax - d["velocity_mps"])))
    grid, spec = angle_music(rd, r_i, d_i, cfg)
    ax.plot(grid, spec, label=f"R={d['range_m']:.1f} m, v={d['velocity_mps']:.1f} m/s")
for t in scene.targets:
    ax.axvline(t.angle_deg, ls="--", c="gray", lw=0.8)
ax.set(xlabel="angle [deg]", ylabel="MUSIC pseudo-spectrum [dB]", ylim=(-40, 2),
       title="Per-detection angle estimation (dashed: truth)")
ax.legend(fontsize=8)
fig.savefig(OUT / "angle_music.png", dpi=130)

# Range profile before/after Doppler processing (shows processing gain)
cube = radar_cube(Y, X)
rp = 10 * np.log10(np.mean(np.abs(np.fft.ifft(cube, n=cfg.n_range_fft, axis=0)) ** 2, axis=(1, 2)))
fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
ax.plot(r_ax, rp - rp.max())
ax.set(xlim=(0, r_lim), xlabel="range [m]", ylabel="dB",
       title="Range profile (slow-time averaged): clutter dominates")
fig.savefig(OUT / "range_profile.png", dpi=130)
print(f"\nSaved figures to {OUT}/")
