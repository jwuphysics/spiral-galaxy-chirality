# The model

The network tells apart clockwise and counterclockwise spiral galaxies. 
It is an ordinary small CNN with one twist. It scores the image, scores 
the mirror image, and subtracts the two scores. A mirror turns a 
clockwise into a counterclockwise galaxy but leaves everything else 
looking the same, so the subtraction removes everything except the winding 
direction. The comments at the top of `net.py` explain this fully.

## Files

- `net.py` is the network. Run `uv run python model/net.py` to see it work.
- `data.py` loads the labeled images and prepares training batches.
- `train.py` trains the network and writes the results page data.
- `test_model.py` runs three quick checks. Start here if something breaks.
- `synthetic.py` draws fake spiral galaxies for testing the code.
- `logpolar_attempt/` is our first design, kept for reference.
- `experiments/` is the design comparison study. See `experiments/STUDY.md`.

## Commands

```sh
uv run python model/test_model.py       # quick checks, under a minute
uv run python model/train.py            # train on your labels
uv run python model/train.py --synthetic  # train on drawn spirals instead
```
