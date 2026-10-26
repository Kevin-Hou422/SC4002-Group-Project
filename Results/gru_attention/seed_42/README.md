# GRU + attention baseline

Base commit: `199b0c111b745b3dbf1a9f093ea05f1a2f23105a`. Seed: 42. CPU. Parameters: 1,019,673.
Selected epoch 39 of 40 by mean validation intent accuracy / slot micro F1.

| Split | Intent accuracy (%) | Slot micro F1 (%) |
| --- | ---: | ---: |
| Validation | 85.49 | 73.18 |
| Test | 83.96 | 70.44 |

The test split was evaluated only after checkpoint selection. Vocabularies use train only.
Slot scoring uses exact (type, span-text) multiset matches, identical to Source/main.ipynb.
All metrics in metrics.json retain full precision; table values are percentages rounded to two decimals.

Training time: 366.7 seconds.

Artifacts: best.pt, config.json, vocabs.json, data_audit.json, environment.json, history.json,
training.log, dev_predictions.jsonl, test_predictions.jsonl, metrics.json, learning_curves.png.

Open `Source/main_GRU_Attention.ipynb` from the repository root for the full implementation.
The checkpoint contains the selected model state, vocabulary, configuration and epoch.
