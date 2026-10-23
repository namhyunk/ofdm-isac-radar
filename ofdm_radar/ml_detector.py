"""Learned target detector on range-Doppler maps (PyTorch).

A small fully-convolutional network predicts a target-center heatmap on the RD map,
trained with a CenterNet-style penalty-reduced focal loss on simulated scenes.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import maximum_filter

from .config import OFDMRadarConfig, Scene, Target
from .processing import radar_cube, range_doppler, static_clutter_removal
from .simulate import simulate_echo


# ---------------------------------------------------------------- data
def random_scene(cfg: OFDMRadarConfig, rng, snr_range=(-25, -5), max_targets=4) -> Scene:
    r_max = 0.9 * min(cfg.cp_limited_range, cfg.max_range)
    v_max = 0.9 * cfg.max_velocity
    targets = [
        Target(rng.uniform(5, r_max), rng.uniform(-v_max, v_max), rng.uniform(-60, 60),
               rcs_db=rng.uniform(-3, 3))
        for _ in range(rng.integers(1, max_targets + 1))
    ]
    clutter = [(rng.uniform(2, r_max), rng.uniform(-70, 70), rng.uniform(10, 25))
               for _ in range(rng.integers(0, 4))]
    return Scene(targets, clutter, snr_db=rng.uniform(*snr_range))


def n_rows(cfg) -> int:
    """Range bins up to the CP-limited range (echoes beyond it are not simulated)."""
    return int(np.ceil(cfg.cp_limited_range / cfg.range_axis()[1]))


def rd_input(cfg, X, Y) -> np.ndarray:
    """Normalized log-power RD map cropped to the CP-limited range, shape (n_rows, n_doppler_fft)."""
    rd = range_doppler(static_clutter_removal(radar_cube(Y, X)), cfg)[: n_rows(cfg)]
    p = 10 * np.log10(np.mean(np.abs(rd) ** 2, axis=2) + 1e-20)
    return ((p - np.median(p)) / (p.std() + 1e-6)).astype(np.float32)


def target_cells(cfg, scene):
    r_axis, v_axis = cfg.range_axis(), cfg.velocity_axis()
    return [(int(np.argmin(np.abs(r_axis - t.range_m))),
             int(np.argmin(np.abs(v_axis - t.velocity_mps)))) for t in scene.targets]


def heatmap(cfg, cells, sigma=(1.5, 1.0)) -> np.ndarray:
    R, D = n_rows(cfg), cfg.n_doppler_fft
    rr, dd = np.meshgrid(np.arange(R), np.arange(D), indexing="ij")
    hm = np.zeros((R, D), np.float32)
    for r, d in cells:
        dd_wrap = np.minimum(np.abs(dd - d), D - np.abs(dd - d))
        hm = np.maximum(hm, np.exp(-((rr - r) ** 2 / (2 * sigma[0] ** 2)
                                     + dd_wrap ** 2 / (2 * sigma[1] ** 2))))
    for r, d in cells:
        hm[r, d] = 1.0
    return hm


def make_dataset(cfg, n, seed=0, **scene_kw):
    rng = np.random.default_rng(seed)
    xs, ys, scenes = [], [], []
    for _ in range(n):
        scene = random_scene(cfg, rng, **scene_kw)
        X, Y = simulate_echo(cfg, scene, rng)
        xs.append(rd_input(cfg, X, Y))
        ys.append(heatmap(cfg, target_cells(cfg, scene)))
        scenes.append(scene)
    return np.stack(xs)[:, None], np.stack(ys)[:, None], scenes


# ---------------------------------------------------------------- model
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, dil=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=dil, dilation=dil, padding_mode="circular")
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return F.gelu(self.bn(self.conv(x)))


class RDDetector(nn.Module):
    """Dilated FCN keeps full RD resolution; circular padding respects Doppler wrap-around."""

    def __init__(self, width=32):
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock(1, width), ConvBlock(width, width, 2), ConvBlock(width, width, 4),
            ConvBlock(width, width, 8), ConvBlock(width, width, 2), ConvBlock(width, width),
        )
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)  # prior: targets are rare

    def forward(self, x):
        return self.head(self.body(x))


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    """CenterNet penalty-reduced focal loss."""
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    pos = target.eq(1).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


def train(model, x, y, epochs=15, batch=32, lr=2e-3, device="cpu", log=print):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * -(-len(x) // batch))
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), batch):
            b = perm[i:i + batch]
            loss = focal_loss(model(xt[b].to(device)), yt[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        log(f"epoch {ep + 1:2d}/{epochs}  loss {tot / len(xt):.4f}")
    return model


@torch.no_grad()
def predict_peaks(model, x, thr=0.3, device="cpu"):
    """Return list of (range_idx, doppler_idx) peaks per sample."""
    model.eval()
    prob = torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()[:, 0]
    out = []
    for p in prob:
        lm = p == maximum_filter(p, size=(5, 3), mode=("nearest", "wrap"))
        out.append([tuple(ix) for ix in np.argwhere(lm & (p > thr))])
    return out
