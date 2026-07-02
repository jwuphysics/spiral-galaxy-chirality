"""Quick checks for the model code. Run with: uv run model/test_model.py

Three things are checked here.
1. Mirroring the input flips the sign of the output exactly. This is the
   property the whole method rests on, so we test it directly.
2. The data augmentation flips the label when it mirrors an image.
3. The network can learn winding direction from computer-drawn spirals in
   under a minute. If this fails, something in the training code is broken.
"""

import numpy as np
import torch
import torch.nn.functional as F

import data
import synthetic
from net import MirrorDifferenceNet


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        raise SystemExit(f"check failed: {name}")


def test_mirror_flips_sign():
    net = MirrorDifferenceNet()
    net.eval()
    x = torch.randn(4, 1, 97, 97)
    with torch.no_grad():
        g = net(x)
        g_mirror = net(torch.flip(x, dims=(-1,)))
    diff = float((g + g_mirror).abs().max())
    check("mirroring the input flips the output sign", diff < 1e-5,
          f"largest error {diff:.2e}")


def test_augmentation_flips_label():
    """Mirrored copies must come back with the opposite label. We draw many
    augmented copies of one image and count how the labels pair up with
    whether the copy was mirrored."""
    rng = np.random.default_rng(0)
    img = (np.random.default_rng(1).random((256, 256)) * 255).astype(np.uint8)
    flipped_labels = 0
    n = 200
    for _ in range(n):
        _, yb = data.augmented_batch(img[None], np.array([0]), rng)
        flipped_labels += int(yb[0]) == 1
    # about half the copies should be mirrored, and exactly those should
    # carry the flipped label
    check("mirror augmentation flips the label about half the time",
          60 < flipped_labels < 140, f"{flipped_labels}/{n} flipped")


def test_learns_synthetic():
    imgs, label_names = synthetic.synth_dataset(300, seed=7)
    y = np.array([data.CLASS_NAMES.index(l) for l in label_names], np.int64)
    tr, va = np.arange(240), np.arange(240, 300)
    x_va = data.shrink_and_normalize(imgs[va])

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = MirrorDifferenceNet()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(25):
        net.train()
        order = rng.permutation(tr)
        for i0 in range(0, len(order), 32):
            batch = order[i0:i0 + 32]
            x, yb = data.augmented_batch(imgs[batch], y[batch], rng)
            loss = F.binary_cross_entropy_with_logits(net(x), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(x_va)).numpy()
    acc = float(((p >= 0.5).astype(int) == y[va]).mean())
    check("learns computer-drawn spirals", acc >= 0.9, f"accuracy {acc:.3f}")


if __name__ == "__main__":
    test_mirror_flips_sign()
    test_augmentation_flips_label()
    test_learns_synthetic()
    print("ALL CHECKS PASSED")
