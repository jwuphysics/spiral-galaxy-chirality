# Results — Spiral Chirality
A static showcase page: run metadata, a log-polar methods sketch, training curves, test accuracy with confusion matrix and P(cw) histogram, and example galleries (correct, misclassified, uncertain).
Data arrives via `results.js`, which `model/train.py` overwrites with `window.RESULTS = {...};` after each run; the copy in-tree is a placeholder, marked as such in `meta.notes`.
Open `index.html` directly in a browser — it works over plain file:// (no fetch, no build step, no libraries).
Charts are hand-drawn inline SVG; example images that do not exist yet degrade to quiet "unavailable" boxes.
Styling is Solarized Light with Baskerville and hairline rules; see `style.css`.
