"""Curate a face-on spiral sample from Galaxy Zoo: Hubble (mwalmsley/gz_hubble on HuggingFace).

Rather than streaming the full 3.2 GB dataset, this queries the HuggingFace
datasets-server /filter API for galaxies passing face-on-spiral vote-fraction
cuts, randomly samples N of them, downloads just those images, center-crops
424 -> 256 px (no resampling), and writes data/images/*.jpg + data/manifest.json.

Usage: python3 scripts/download_gz_hubble.py [--n 1600] [--seed 42] [--out data]
"""

import argparse
import io
import json
import random
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

DATASET = "mwalmsley/gz_hubble"
API = "https://datasets-server.huggingface.co/filter"
PAGE = 100
CROP = 256

CUTS = {
    "smooth-or-featured-hubble_features_fraction": (">", 0.5),
    "disk-edge-on-hubble_no_fraction": (">", 0.75),
    "has-spiral-arms-hubble_yes_fraction": (">", 0.8),
    "has-spiral-arms-hubble_total-votes": (">=", 10),
    "smooth-or-featured-hubble_artifact_fraction": ("<", 0.2),
    "clumpy-appearance-hubble_yes_fraction": ("<", 0.3),
}

WHERE = " AND ".join(f'"{col}"{op}{val}' for col, (op, val) in CUTS.items())


def api_get(params, retries=4):
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            print(f"  retrying page fetch ({e})")


def fetch_candidates():
    """Page through the filter API, returning all matching rows."""
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        d = api_get(
            {"dataset": DATASET, "config": "default", "split": "train",
             "where": WHERE, "offset": offset, "limit": PAGE}
        )
        total = d["num_rows_total"]
        rows.extend(r["row"] for r in d["rows"])
        offset += len(d["rows"])
        print(f"  fetched {offset}/{total} candidate rows")
        if not d["rows"]:
            break
    return rows


def clean_id(id_str):
    """gz_hubble id_str values end in '.jpg'; strip it."""
    return id_str[:-4] if id_str.endswith(".jpg") else id_str


def save_image(row, images_dir, retries=3):
    gid = clean_id(row["id_str"])
    dest = images_dir / f"{gid}.jpg"
    if dest.exists():
        return gid, True
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(row["image"]["src"], timeout=60) as r:
                im = Image.open(io.BytesIO(r.read())).convert("RGB")
            w, h = im.size
            left, top = (w - CROP) // 2, (h - CROP) // 2
            im.crop((left, top, left + CROP, top + CROP)).save(dest, quality=92)
            return gid, True
        except Exception:
            if attempt == retries - 1:
                return gid, False
            time.sleep(1 + attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = ap.parse_args()

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"querying candidates: {WHERE}")
    rows = fetch_candidates()
    print(f"{len(rows)} candidates pass cuts")

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    print(f"sampled {len(sample)} (seed={args.seed}); downloading...")

    ok, failed = set(), []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(save_image, row, images_dir): row for row in sample}
        for i, fut in enumerate(as_completed(futures), 1):
            gid, success = fut.result()
            (ok.add(gid) if success else failed.append(gid))
            if i % 100 == 0:
                print(f"  {i}/{len(sample)} downloaded")

    items = [
        {
            "id": clean_id(row["id_str"]),
            "file": f"images/{clean_id(row['id_str'])}.jpg",
            "ra": row["ra"],
            "dec": row["dec"],
            "features_frac": round(row["smooth-or-featured-hubble_features_fraction"], 4),
            "not_edgeon_frac": round(row["disk-edge-on-hubble_no_fraction"], 4),
            "spiral_frac": round(row["has-spiral-arms-hubble_yes_fraction"], 4),
            "spiral_votes": row["has-spiral-arms-hubble_total-votes"],
        }
        for row in sorted(sample, key=lambda r: r["id_str"])
        if clean_id(row["id_str"]) in ok
    ]

    manifest = {
        "dataset": DATASET,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cuts": {col: f"{op} {val}" for col, (op, val) in CUTS.items()},
        "n_candidates": len(rows),
        "seed": args.seed,
        "image_size": CROP,
        "crop": f"center {CROP} of 424 (native pixels)",
        "items": items,
    }
    manifest_path = args.out / "manifest.json"
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(manifest_path)

    print(f"wrote {manifest_path} with {len(items)} items ({len(failed)} failed downloads)")
    if failed:
        print("failed ids:", failed[:20], "..." if len(failed) > 20 else "")


if __name__ == "__main__":
    main()
