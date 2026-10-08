# Trained model files

This folder is **empty on purpose**. No trained weights are shipped: they must be generated from the dataset
by running the training command (see the main README, section 7).

After training you will find here:

| File | What it is |
|---|---|
| `signbridge_head.pt` | trained BiLSTM + Attention + Dense head (with all settings needed to reload it) |
| `classes.json` | class id -> sentence mapping |
| `history.json` | loss / accuracy for every epoch |
| `splits.json` | which videos went into train / validation / test |
| `metrics.json` | real test accuracy, precision, recall, F1 |
| `confusion_matrix.csv` / `.png` | real confusion matrix |
| `latency.json` | real measured inference latency (after `python -m training.benchmark_latency`) |
| `feature_cache/` | cached CNN features (can be deleted; they are re-created) |
