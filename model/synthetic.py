"""Procedural face-on spiral galaxies, 256x256 grayscale uint8. numpy-only.

Chirality convention (matches the labeler and Galaxy Zoo 1): "ccw" is the
S-SHAPED pattern (GZ1 "S-wise / anti-clockwise"), "cw" the Z-shaped one.
NOTE the subtlety: GZ1's "anticlockwise" names the apparent SPIN assuming
trailing arms. Traced outward from the center, an S-shaped arm's on-screen
azimuth DECREASES (sweeps clockwise) — verified visually against rendered
letters. Screen y points down, so the on-screen-CCW azimuth is
phi = atan2(cy - y, x - cx), and a "ccw" (S-wise) arm obeys
r = a * exp(-tan(pitch) * phi), i.e. b < 0.
"""

import numpy as np

SIZE = 256
_TWO_PI = 2.0 * np.pi


def _grids(size):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2.0
    dx = xx - cx
    dy = cy - yy  # up on screen
    r = np.hypot(dx, dy)
    phi = np.mod(np.arctan2(dy, dx), _TWO_PI)
    log_r = np.log(np.maximum(r, 0.5))
    return r, phi, log_r


def synth_dataset(n, seed, size=SIZE, chunk=200):
    """Returns (images uint8 [n, size, size], labels list of "ccw"/"cw")."""
    rng = np.random.default_rng(seed)
    r, phi, log_r = _grids(size)
    r_, phi_, logr_ = r[None], phi[None], log_r[None]
    fy = np.fft.fftfreq(size).astype(np.float32)
    fx = np.fft.rfftfreq(size).astype(np.float32)
    f2 = (fy[:, None] ** 2 + fx[None, :] ** 2)[None]

    imgs = np.empty((n, size, size), np.uint8)
    labels = []
    for i0 in range(0, n, chunk):
        c = min(chunk, n - i0)
        col = lambda a: a.astype(np.float32)[:, None, None]
        m = rng.integers(2, 4, c)                       # 2-3 arms
        pitch = np.deg2rad(rng.uniform(10.0, 35.0, c))
        is_ccw = rng.random(c) < 0.5
        b = np.where(is_ccw, -1.0, 1.0) * np.tan(pitch)  # b<0 = ccw (S-wise) on screen
        pa = rng.uniform(0.0, _TWO_PI, c)               # position angle
        r_disk = rng.uniform(38.0, 68.0, c)
        width = rng.uniform(2.5, 5.0, c)                # arm cross-section sigma, px
        a_arm = rng.uniform(0.9, 1.6, c)
        a_disk = rng.uniform(0.15, 0.35, c)
        a_bulge = rng.uniform(1.2, 2.2, c)
        s_bulge = rng.uniform(4.0, 9.0, c)
        psf = rng.uniform(1.0, 2.5, c)
        sky = rng.uniform(1.0, 3.0, c)

        period = _TWO_PI / m
        phase = phi_ - logr_ / col(b) - col(pa)
        delta = np.mod(phase + col(period) / 2, col(period)) - col(period) / 2
        d_perp = r_ * delta * col(np.cos(pitch))        # distance to nearest arm
        img = (col(a_arm) * np.exp(-0.5 * (d_perp / col(width)) ** 2)
               * np.exp(-r_ / col(r_disk)) * (1.0 - np.exp(-((r_ / 5.0) ** 2))))
        img += col(a_disk) * np.exp(-r_ / col(r_disk))
        img += col(a_bulge) * np.exp(-0.5 * (r_ / col(s_bulge)) ** 2)
        img *= 190.0 / img.max(axis=(1, 2), keepdims=True)

        # Gaussian PSF via FFT (transfer function exp(-2 pi^2 sigma^2 f^2))
        img = np.fft.irfft2(np.fft.rfft2(img)
                            * np.exp(-2.0 * np.pi ** 2 * col(psf) ** 2 * f2),
                            s=(size, size))
        img += rng.standard_normal(img.shape, dtype=np.float32) * col(sky)
        img += (rng.standard_normal(img.shape, dtype=np.float32)
                * 0.35 * np.sqrt(np.maximum(img, 0.0)))
        imgs[i0:i0 + c] = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        labels += ["ccw" if f else "cw" for f in is_ccw]
    return imgs, labels
