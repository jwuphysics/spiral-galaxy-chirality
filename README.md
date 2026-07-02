# Spiral chirality

The arms of a spiral galaxy wind in one of two directions. Seen from Earth, some galaxies trace the letter S and some trace the letter Z. This project trains a small neural network to tell the two apart, using Hubble telescope images that you label by hand in a browser tool.

## The idea in one paragraph

If you look at a photo of an S galaxy in a mirror, you see a Z galaxy. Everything else in the photo keeps its character in the mirror. A blob stays a blob, noise stays noise, and a tilted disk stays a tilted disk. The network uses this fact directly. It gives each image one score. We score the image, we score its mirror image, and we subtract. Anything that looks the same in a mirror cancels out of the subtraction. The winding direction is the only thing that survives, so it is the only thing the network can use to decide. Without this trick the exact same network learns nothing from our data, and stays at coin-flip accuracy. With it, the network reached about 88 percent accuracy in our tests with only 68 labeled galaxies.

## How to use it

The project uses [uv](https://docs.astral.sh/uv/) to manage Python. Run `uv sync` once, then follow these steps.

1. Download the data.

   ```sh
   uv run scripts/download_gz_hubble.py
   ```

   This picks about 1,600 galaxies from the Galaxy Zoo Hubble catalog that volunteers voted to be face-on spirals. It saves them as small images in `data/images/` and writes a list of them to `data/manifest.json`.

2. Label galaxies.

   ```sh
   uv run labeler/serve.py
   ```

   Then open http://localhost:8801/labeler/ in a browser. Look at each galaxy and press a key:

   - `1` if the arms trace the letter S
   - `2` if the arms trace the letter Z
   - `3` to discard the image, e.g., if you cannot see the arms
   - `0` to clear the label on the current galaxy
   - left and right arrows to move around

   Every keypress saves your work to `labeler/labels.json`, so you can stop and come back at any time. Training needs at least 40 labeled galaxies, but more is better.

3. Train the network.

   ```sh
   uv run model/train.py
   ```

   This takes a few minutes. Add `--synthetic` to train on computer-drawn spirals instead, which is a quick way to check that the code works before you have labels.

4. Look at the results. Open `results/index.html` in a browser. The page shows the training curves, the accuracy on held-out galaxies, and examples of what the network got right and wrong.

## Naming the two directions

We call the S shape "ccw" (counterclockwise) and the Z shape "cw" (clockwise) in the code and in the data files. These are just names. Judge each galaxy only by whether its arms trace an S or a Z, and use the two colored glyphs at the bottom of the labeler page as your reference.

## What is in each folder

- `data/` holds the downloaded galaxy images and their list.
- `labeler/` is the browser tool for labeling, plus the labels you make.
- `model/` is the network (`net.py`), the data preparation (`data.py`), the training script (`train.py`), and quick checks (`test_model.py`).
- `model/logpolar_attempt/` is our first design, kept for reference. It unwrapped each galaxy around its center so spiral arms became straight lines. It worked perfectly on computer-drawn spirals but failed on real galaxies, which are noisy and tilted.
- `model/experiments/` is the study that compared network designs and found that the mirror trick is what makes learning possible. `STUDY.md` in that folder summarizes it.
- `results/` is the results web page. Training writes `results.js` there, and the page reads it. The page works when opened straight from disk, with no server.
- `scripts/` holds the data download script.
