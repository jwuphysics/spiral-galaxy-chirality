"""Load the labeled galaxy images and prepare them for training.

What happens to each image
--------------------------
The images on disk are 256 by 256 pixel cutouts with the galaxy in the
center. Before an image goes into the network we shrink it to 97 by 97
pixels (small images train faster and the winding direction is still easy
to see) and we rescale its pixel values so that every image has a similar
brightness range. We rescale by subtracting the middle pixel value (the
median) and dividing by the spread (the standard deviation). Without this
step, a bright galaxy and a faint galaxy would look like different problems
to the network.

Why 97 and not 96: we shrink the image by half twice inside the network,
and odd sizes keep the center pixel exactly in the center each time.

Training tricks (data augmentation)
-----------------------------------
We have very few labeled galaxies, so during training we show the network
altered copies of them. Each copy is rotated by a random angle and shifted
by a few pixels. Rotating or shifting a galaxy does not change its winding
direction, so the label stays the same. Half the time we also mirror the
copy, and then we FLIP the label, because a mirror turns an S galaxy into a
Z galaxy. This mirror trick doubles our data for free and also keeps the
two classes perfectly balanced during training.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLASS_NAMES = ["ccw", "cw"]  # index 0 = S shape, index 1 = Z shape
MODEL_DIR = Path(__file__).resolve().parent
NET_INPUT_SIZE = 97


class DataError(RuntimeError):
    pass


def load_labeled(root):
    """Read the image files that the user labeled as S (ccw) or Z (cw).

    Returns three things: the images as an array of shape [count, 256, 256]
    with whole number pixel values from 0 to 255, the labels as an array of
    0s and 1s, and the list of galaxy id strings.
    """
    root = Path(root)
    manifest_path = root / "data" / "manifest.json"
    labels_path = root / "labeler" / "labels.json"
    if not manifest_path.exists():
        raise DataError(f"{manifest_path} not found. Run the download script "
                        "first (see the README).")
    if not labels_path.exists():
        raise DataError(f"{labels_path} not found. Label some galaxies first "
                        "with the labeler (see the README).")
    manifest = json.loads(manifest_path.read_text())
    labels = json.loads(labels_path.read_text())["labels"]

    items = [it for it in manifest["items"]
             if labels.get(str(it["id"]), {}).get("label") in CLASS_NAMES]
    if len(items) < 40:
        raise DataError(f"Only {len(items)} galaxies are labeled S or Z. "
                        "Please label at least 40.")

    ids = [str(it["id"]) for it in items]
    y = np.array([CLASS_NAMES.index(labels[i]["label"]) for i in ids], np.int64)
    imgs = np.stack([
        np.asarray(Image.open(root / "data" / it["file"]).convert("L"), np.uint8)
        for it in items])
    files = {str(it["id"]): "../data/" + it["file"] for it in items}
    return imgs, y, ids, files


def stratified_split(y, seed, fracs=(0.70, 0.15, 0.15)):
    """Split the data into training, validation, and test parts.

    We split each class separately so that both classes appear in every
    part in the same proportion. The random shuffle is fixed by the seed,
    so the same seed always produces exactly the same split. This is
    important: the test part must stay the same across runs, so that nobody
    accidentally tunes the model on it.
    """
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_tr = int(round(fracs[0] * len(idx)))
        n_va = int(round(fracs[1] * len(idx)))
        parts[0].extend(idx[:n_tr])
        parts[1].extend(idx[n_tr:n_tr + n_va])
        parts[2].extend(idx[n_tr + n_va:])
    return tuple(np.sort(np.asarray(p, np.int64)) for p in parts)


def shrink_and_normalize(imgs):
    """Turn raw images into network input.

    Takes an array of shape [count, 256, 256] with any pixel scale.
    Shrinks each image to 97 by 97 with a gentle blur so details do not
    turn jagged. Then rescales each image by subtracting its median pixel
    value and dividing by its standard deviation. Returns a tensor of shape
    [count, 1, 97, 97].
    """
    x = torch.as_tensor(np.ascontiguousarray(imgs), dtype=torch.float32)[:, None]
    x = F.interpolate(x, size=(NET_INPUT_SIZE, NET_INPUT_SIZE),
                      mode="bilinear", antialias=True)
    flat = x.flatten(1)
    med = flat.median(dim=1).values
    std = flat.std(dim=1, unbiased=False).clamp_min(1e-3)
    return (x - med[:, None, None, None]) / std[:, None, None, None]


def augmented_batch(imgs_u8, ys, rng):
    """Make one training batch of randomly altered image copies.

    For each image we pick a random rotation angle, a small random shift,
    and a coin flip that decides whether to mirror it. Mirroring also flips
    the label. We do the rotation on the full 256 pixel image, because
    rotating after shrinking would blur the arms too much. At the end we
    multiply each image by a random brightness factor between 0.8 and 1.25
    so the network cannot rely on exact brightness values.
    """
    arrays, labels, gains = [], [], []
    for img, y in zip(imgs_u8, ys):
        angle = float(rng.uniform(0.0, 360.0))
        tx, ty = (int(v) for v in rng.integers(-8, 9, size=2))
        mirror = bool(rng.random() < 0.5)
        fill = int(np.median(img))
        rotated = Image.fromarray(img).rotate(
            angle, resample=Image.BILINEAR, translate=(tx, ty), fillcolor=fill)
        arr = np.asarray(rotated, np.float32)
        y = int(y)
        if mirror:
            arr = np.ascontiguousarray(arr[:, ::-1])
            y = 1 - y
        arrays.append(arr)
        labels.append(y)
        gains.append(float(rng.uniform(0.8, 1.25)))
    x = shrink_and_normalize(np.stack(arrays))
    x = x * torch.tensor(gains, dtype=torch.float32)[:, None, None, None]
    return x, torch.tensor(labels, dtype=torch.float32)
