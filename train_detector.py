"""Train a CNN target detector on simulated range-Doppler maps and compare it to CA-CFAR.

    python train_detector.py                 # ~3 min on a 2-core CPU, seconds on a GPU
    python train_detector.py --n-train 8000 --epochs 25 --width 32
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ofdm_radar import OFDMRadarConfig
from ofdm_radar.ml_detector import (RDDetector, make_dataset, n_rows, predict_peaks,
                                    random_scene, rd_input, target_cells, train)
from ofdm_radar.processing import (ca_cfar_2d, cluster_peaks, radar_cube, range_doppler,
                                   static_clutter_removal)
from ofdm_radar.simulate import simulate_echo

ap = argparse.ArgumentParser()
ap.add_argument("--n-train", type=int, default=2000)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--n-test", type=int, default=100, help="frames per SNR point")
ap.add_argument("--width", type=int, default=16)
ap.add_argument("--cnn-thr", type=float, default=0.3)
ap.add_argument("--pfa", type=float, default=1e-4)
args = ap.parse_args()

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# Smaller frame than the demo so the dataset is cheap to simulate.
cfg = OFDMRadarConfig(n_sc=512, n_sym=32, n_range_fft=1024, n_doppler_fft=64)
pr, pd = cfg.pad_factors
print(cfg.summary())
print(f"RD input size: {n_rows(cfg)} x {cfg.n_doppler_fft}, device: {device}\n")

x, y, _ = make_dataset(cfg, args.n_train, seed=1)
model = train(RDDetector(args.width), x, y, epochs=args.epochs, device=device)
torch.save(model.state_dict(), OUT.parent / "rd_detector.pt")


def match(pred, truth, tol=(2 * pr, pd)):
    """Greedy matching with Doppler wrap-around. Returns (hits, false alarms)."""
    D = cfg.n_doppler_fft
    used, hits = set(), 0
    for tr, td in truth:
        for j, (r, d) in enumerate(pred):
            dd = min(abs(d - td), D - abs(d - td))
            if j not in used and abs(r - tr) <= tol[0] and dd <= tol[1]:
                used.add(j)
                hits += 1
                break
    return hits, len(pred) - len(used)


def cfar_peaks(X, Y):
    rd = range_doppler(static_clutter_removal(radar_cube(Y, X)), cfg)[: n_rows(cfg)]
    p = np.mean(np.abs(rd) ** 2, axis=2)
    mask, _ = ca_cfar_2d(p, guard=(3 * pr, 2 * pd), train=(6 * pr, 4 * pd), pfa=args.pfa)
    return [tuple(ix) for ix in cluster_peaks(p, mask, size=(4 * pr + 1, 3 * pd + 1))]


snrs = np.arange(-30, -4, 3)
res = {"cfar": [], "cnn": []}
print(f"\n{'SNR':>5} | {'CFAR Pd':>8} {'FA/frm':>7} | {'CNN Pd':>8} {'FA/frm':>7}")
for snr in snrs:
    rng = np.random.default_rng(1000 + int(snr))
    stats = {"cfar": [0, 0], "cnn": [0, 0]}
    n_tgt = 0
    xs, truths, cfar_pred = [], [], []
    for _ in range(args.n_test):
        scene = random_scene(cfg, rng, snr_range=(snr, snr))
        X, Y = simulate_echo(cfg, scene, rng)
        xs.append(rd_input(cfg, X, Y))
        truths.append(target_cells(cfg, scene))
        cfar_pred.append(cfar_peaks(X, Y))
        n_tgt += len(scene.targets)
    cnn_pred = predict_peaks(model, np.stack(xs)[:, None], thr=args.cnn_thr, device=device)
    for t, pc, pn in zip(truths, cfar_pred, cnn_pred):
        for name, pred in (("cfar", pc), ("cnn", pn)):
            h, fa = match(pred, t)
            stats[name][0] += h
            stats[name][1] += fa
    for name in res:
        res[name].append((stats[name][0] / n_tgt, stats[name][1] / args.n_test))
    print(f"{snr:5d} | {res['cfar'][-1][0]:8.3f} {res['cfar'][-1][1]:7.3f} | "
          f"{res['cnn'][-1][0]:8.3f} {res['cnn'][-1][1]:7.3f}")

fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for name, lbl in (("cfar", f"CA-CFAR (Pfa={args.pfa:g})"), ("cnn", f"CNN (thr={args.cnn_thr})")):
    r = np.array(res[name])
    axs[0].plot(snrs, r[:, 0], "o-", label=lbl)
    axs[1].plot(snrs, r[:, 1], "o-", label=lbl)
axs[0].set(xlabel="per-element SNR [dB]", ylabel="probability of detection", title="Detection rate")
axs[1].set(xlabel="per-element SNR [dB]", ylabel="false alarms per frame", title="False alarms")
for a in axs:
    a.grid(alpha=0.3)
    a.legend()
fig.savefig(OUT / "cnn_vs_cfar.png", dpi=130)
print(f"\nSaved {OUT / 'cnn_vs_cfar.png'}")
