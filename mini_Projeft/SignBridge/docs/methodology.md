# Methodology and research documentation

## Problem statement
Communication between Indian Sign Language users and non-sign-language users is difficult because they do not
share a common communication medium. SignBridge aims to reduce this barrier using a camera- and
microphone-based two-way communication assistant.

## Existing systems and limitations
Existing work covers CNN-based ISL recognition, CNN+LSTM, MediaPipe+LSTM, continuous and online recognition,
sign-to-text/speech, speech/text-to-sign, bidirectional systems and transformer-based translation. We do **not**
claim any of these individual functions is new. (Add your own literature survey here: for each paper give title,
year, dataset, model, result and limitation - only from papers you have actually read.)

Limitations we address in the design (short names):
1. **Continuity** - limited handling of continuous sign sequences.
2. **Temporal** - difficulty understanding temporal context.
3. **Latency** - real-time processing can introduce delay.
4. **Vocabulary** - limited vocabulary / dataset coverage.
5. **Non-Manual** - limited handling of facial expressions and body movement.

Honest status of each in this prototype:
| Limitation | What we do | What remains open |
|---|---|---|
| Continuity | sentence-level clips and a sliding window | not continuous translation; closed set |
| Temporal | BiLSTM + attention over sampled frames | evaluate with/without attention |
| Latency | latency is measured (`benchmark_latency`, UI) | depends on hardware; not claimed real-time |
| Vocabulary | dictionary from the dataset CSV | only ~100 sentences |
| Non-Manual | full-frame CNN features; optional face landmarks | not explicitly modelled; optional module untested on real data |

## Research gap and contribution
Existing approaches address different components such as sign recognition, temporal modelling, speech/text
conversion, or sign representation. SignBridge integrates these components into a practical two-way
communication prototype and evaluates the recognition pipeline using accuracy and end-to-end latency.
We do **not** claim the components, or the CNN + BiLSTM + Attention combination, have never been built before.

## Method summary
1. Read dataset (CSV + videos) dynamically; audit it.
2. Sample `NUM_FRAMES` frames per video; resize/normalise.
3. Frozen MobileNetV2 features (optionally + MediaPipe landmarks).
4. BiLSTM -> temporal attention -> dense -> softmax (100-way sentence classification).
5. Train with Adam, cross-entropy, early stopping and best-checkpoint saving.
6. Evaluate on a held-out test split: accuracy, macro/weighted precision, recall, F1, confusion matrix.
7. Measure inference latency separately (server side) and end-to-end in the UI.

## Suggested experiments for your report (all optional, all use the provided code)
* Attention vs no attention: `USE_ATTENTION=false`.
* Number of frames: `NUM_FRAMES` = 16 / 24 / 30 / 45.
* Backbone: `mobilenet_v2` vs `resnet18` vs `efficientnet_b0`.
* With vs without MediaPipe (only if the footage supports it).
* Signer-independent vs random split (only if signer IDs exist) - report both honestly.

## Threats to validity (write these in the report)
* Small dataset, few signers: test accuracy may not generalise to new signers.
* Random splits can leak the same signer into train and test.
* Dataset videos differ from live webcam input.
* A closed-set classifier always outputs a class; confidence thresholds only partly compensate.
* Latency numbers are hardware-specific.
