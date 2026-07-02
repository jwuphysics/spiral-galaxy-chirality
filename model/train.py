"""Train the mirror-difference CNN to classify spiral winding direction.

How to run it
-------------
    uv run model/train.py                 train on your hand-labeled galaxies
    uv run model/train.py --synthetic     train on computer-drawn spirals
                                          (a quick check that the code works)

What it does
------------
1. Loads the labeled galaxies and splits them into three fixed parts:
   70 percent for training, 15 percent for choosing nothing (we keep it to
   watch progress), and 15 percent held out for the final test. The split
   depends only on the seed, so it is the same every run.
2. Trains the network for a fixed number of passes over the training data.
   Each pass shows every training galaxy once, randomly rotated, shifted,
   and possibly mirrored (with the label flipped to match).
3. Measures accuracy on the held-out test part exactly once, at the end.
4. Writes everything the results web page needs into the results folder:
   the training curves, the test numbers, and example images.

We report the network from the LAST training pass rather than picking the
pass with the best validation score. The validation part has only about a
dozen galaxies, so picking the best pass by it would mostly reward luck.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import data
import synthetic
from net import MirrorDifferenceNet

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parent


def evaluate(net, x, y, device):
    """Run the network on a prepared batch and return loss, accuracy, and
    the predicted probability that each galaxy is Z shaped (clockwise)."""
    net.eval()
    with torch.no_grad():
        logits = net(x.to(device)).cpu()
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.as_tensor(y, dtype=torch.float32))
        p_cw = torch.sigmoid(logits).numpy()
    acc = float(((p_cw >= 0.5).astype(int) == np.asarray(y)).mean())
    return float(loss), acc, p_cw


def displayable(x97):
    """Turn one normalized 97 by 97 network input into an image we can save.

    The normalized pixel values are roughly between -2 and +10. We clip
    that range and stretch it to 0 to 255 so the picture is easy to look at.
    """
    a = np.clip(x97, -1.5, 6.0)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return (a * 255).astype(np.uint8)


def pick_examples(ids, y_true, p_cw, k=12):
    """Choose interesting galaxies for the results page galleries.

    We sort by confidence, which is how far the predicted probability is
    from 0.5. We return three groups: confidently correct predictions,
    all the wrong predictions, and the least confident predictions.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(p_cw) >= 0.5).astype(int)
    conf = np.abs(np.asarray(p_cw) - 0.5)
    correct = np.flatnonzero(pred == y_true)
    wrong = np.flatnonzero(pred != y_true)
    groups = {
        "correct_confident": correct[np.argsort(-conf[correct])][:k],
        "incorrect": wrong[np.argsort(-conf[wrong])][:k],
        "uncertain": np.argsort(conf)[:k],
    }
    return groups, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="train on computer-drawn spirals instead of real data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # ---- load the data --------------------------------------------------
    if args.synthetic:
        imgs, label_names = synthetic.synth_dataset(2000, args.seed)
        y = np.array([data.CLASS_NAMES.index(l) for l in label_names], np.int64)
        ids = [f"syn_{i:05d}" for i in range(len(y))]
        files = None
        dataset_name = "synthetic"
    else:
        try:
            imgs, y, ids, files = data.load_labeled(args.data_root)
        except data.DataError as e:
            sys.exit(str(e))
        dataset_name = "mwalmsley/gz_hubble"

    tr, va, te = data.stratified_split(y, args.seed)
    print(f"{len(tr)} training, {len(va)} validation, {len(te)} test galaxies")

    # The validation and test images are prepared once, with no random
    # alterations, so the numbers we track are stable.
    x_va = data.shrink_and_normalize(imgs[va])
    x_te = data.shrink_and_normalize(imgs[te])

    net = MirrorDifferenceNet().to(device)
    print(f"network has {net.n_params} learned numbers, training on {device}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ---- training loop ---------------------------------------------------
    history = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(tr)
        losses, hits, count = 0.0, 0, 0
        for i0 in range(0, len(order), args.bs):
            batch = order[i0:i0 + args.bs]
            x, yb = data.augmented_batch(imgs[batch], y[batch], rng)
            x, yb = x.to(device), yb.to(device)
            logits = net(x)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses += float(loss.detach()) * len(batch)
            hits += int(((logits >= 0) == (yb >= 0.5)).sum())
            count += len(batch)
        sched.step()
        val_loss, val_acc, _ = evaluate(net, x_va, y[va], device)
        history.append({"epoch": epoch,
                        "train_loss": round(losses / count, 4),
                        "train_acc": round(hits / count, 4),
                        "val_loss": round(val_loss, 4),
                        "val_acc": round(val_acc, 4)})
        if epoch % 10 == 0 or epoch == 1:
            h = history[-1]
            print(f"epoch {epoch:3d}/{args.epochs}  "
                  f"train loss {h['train_loss']:.4f} acc {h['train_acc']:.4f}  |  "
                  f"val loss {h['val_loss']:.4f} acc {h['val_acc']:.4f}")

    ckpt_dir = MODEL_DIR / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    # synthetic runs get their own file so they never overwrite a real run
    ckpt_path = ckpt_dir / ("final_synthetic.pt" if args.synthetic else "final.pt")
    torch.save(net.state_dict(), ckpt_path)

    # ---- the one and only test measurement -------------------------------
    test_loss, test_acc, p_cw = evaluate(net, x_te, y[te], device)
    y_te = y[te]
    pred = (p_cw >= 0.5).astype(int)
    confusion = [[int(((y_te == t) & (pred == p)).sum()) for p in (0, 1)]
                 for t in (0, 1)]
    print(f"test accuracy {test_acc:.4f} on {len(te)} galaxies  "
          f"confusion (rows are the true class, S then Z) {confusion}")

    # ---- save everything the results page needs --------------------------
    out = args.out
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    test_ids = [ids[i] for i in te]
    groups, _ = pick_examples(test_ids, y_te, p_cw)

    examples = {}
    for name, idxs in groups.items():
        entries = []
        for j in idxs:
            gid = test_ids[j]
            Image.fromarray(displayable(x_te[j, 0].numpy())).save(
                out / "inputs" / f"{gid}.png")
            if args.synthetic:
                img_dir = out / "synthetic_images"
                img_dir.mkdir(exist_ok=True)
                Image.fromarray(imgs[te[j]]).save(img_dir / f"{gid}.png")
                file_rel = f"synthetic_images/{gid}.png"
            else:
                file_rel = files[gid]
            entries.append({"id": gid,
                            "file": file_rel,
                            "input_file": f"inputs/{gid}.png",
                            "true": data.CLASS_NAMES[int(y_te[j])],
                            "pred": data.CLASS_NAMES[int(pred[j])],
                            "p_cw": round(float(p_cw[j]), 4)})
        examples[name] = entries

    results = {
        "meta": {"run_name": "mirror-difference-cnn",
                 "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "dataset": dataset_name,
                 "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                 "n_params": net.n_params,
                 "epochs": args.epochs, "batch_size": args.bs,
                 "lr": args.lr, "seed": args.seed,
                 "class_names": data.CLASS_NAMES,
                 "notes": ("trained on synthetic spirals" if args.synthetic else
                           "a plain CNN scored on the image minus its mirror image")},
        "history": history,
        "test": {"accuracy": round(test_acc, 4), "n": len(te),
                 "confusion": confusion,
                 "p_cw": [round(float(p), 4) for p in p_cw]},
        "examples": examples,
    }
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "results.js").write_text("window.RESULTS = " + json.dumps(results) + ";")
    print(f"wrote {out / 'results.json'} and results.js; "
          f"checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
