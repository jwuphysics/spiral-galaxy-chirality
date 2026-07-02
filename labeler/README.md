# Spiral Chirality · Labeler

Run from the project root: `python3 labeler/serve.py` (default port 8801),
then open http://localhost:8801/labeler/ — it loads `data/manifest.json`.

Keys: `1` ccw (S-wise, counterclockwise) · `2` cw (Z-wise, clockwise) ·
`3` discard · `0` clear label · `←`/`→` navigate. Every keypress is saved
immediately and the view advances to the next unlabeled galaxy.

Labels go to `labeler/labels.json` (atomic writes; safe to interrupt).
Testing: append `?manifest=testdata/manifest.json&labels=labels_test.json`
to use the procedural fixtures in `labeler/testdata/` instead of real data.
