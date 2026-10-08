# Dataset notes - ISL-CSLTR (primary dataset)

This file separates **what was verified from a source** from **what is not verified**. Re-check everything after
you download the data (`python scripts\audit_dataset.py`).

## Verified from the original Mendeley Data page (https://data.mendeley.com/datasets/kcmpdxky7p/1)
* Name: *ISL-CSLTR: Indian Sign Language Dataset for Continuous Sign Language Translation and Recognition*
* Authors: Elakkiya R, Natarajan B. Published as Mendeley Data, version 1, January 2021 (DOI 10.17632/kcmpdxky7p.1).
* Content described: 700 annotated videos of 100 spoken-language sentences, plus word-level images
  (1036) and sentence-level frames (18,863); labels through a CSV file; recorded with a DSLR camera under
  varied angles, backgrounds and lighting.

## Not verified (unknown until you run the audit)
* **Licence.** The page we could read showed no licence text. A third-party Kaggle mirror lists CC BY-SA 4.0,
  which is **not** evidence for the original. Read the licence on the Mendeley page yourself and cite the dataset.
* Exact number of videos (a secondary source mentions a different number), videos per sentence.
* Signer count: the description says 7 signers, but its text lists two native signers and four volunteers (6).
* Video format, resolution, FPS, duration, whether the face and both hands are visible, whether signer IDs exist
  in the CSV, official train/validation/test split (none found), total size on disk.

## Comparison with the other candidates (facts from their papers/pages)
| | ISL-CSLTR | iSign | ISLTranslate |
|---|---|---|---|
| Size | 700 videos (per Mendeley) | 118K+ video-sentence/phrase pairs (arXiv 2407.05404) | 31,222 pairs (ACL Findings 2023) |
| Labels | sentence text (CSV), 100 sentences | English translations + pose files | English translations |
| Licence | not verified | CC-BY-NC-SA-4.0 (Hugging Face page) | not verified |
| Fits a softmax classifier? | Yes (100 classes) | No - translation task | No - translation task |
| Laptop training | likely yes | no | no |

Why ISL-CSLTR: it is the only candidate whose labels match a fixed-class softmax model, and it is small enough
for a student laptop. Its weakness is the small size and the limited number of signers. iSign or ISLTranslate would
be the route for a future translation-style (sequence-to-sequence) system.

## What the code assumes (and how it checks)
* Videos + one CSV linking file names to sentences. Column names are auto-detected and can be overridden in
  `config.json` (`CSV_FILE`, `VIDEO_COLUMN`, `LABEL_COLUMN`, `SIGNER_COLUMN`).
* If a signer-like column exists the split is signer-independent, otherwise it is random and a warning is stored.
