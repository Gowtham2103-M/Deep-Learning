# Where to put the ISL-CSLTR dataset

The dataset is **not included** in this project (it is large, and we do not have the right to redistribute it).

## Steps
1. Download **ISL-CSLTR: Indian Sign Language Dataset for Continuous Sign Language Translation and Recognition**
   (authors: Elakkiya R and Natarajan B, Mendeley Data, 2021, DOI 10.17632/kcmpdxky7p.1):
   https://data.mendeley.com/datasets/kcmpdxky7p/1
2. **Read the licence shown on that page** and cite the dataset in your report. (We could not verify the
   licence text ourselves - see `docs/dataset.md`.)
3. Unzip it so that this folder exists and contains the videos and the CSV file(s):

```
SignBridge/
  dataset/
    ISL_CSLRT_Corpus/        <- put the unzipped dataset here (the name can be different, see below)
       ... videos (.mp4 / .avi / ...) in any sub-folders ...
       ... CSV file(s) with the sentence labels ...
```

The project searches all sub-folders, so the exact inner structure does not matter.

## Different folder name or location?
Copy `config.example.json` to `config.json` in the project folder and change:
```json
{ "DATASET_PATH": "dataset/MyFolderName" }
```
An absolute path such as `D:/data/ISL` also works.

## Check that it was found
From the `backend` folder (virtual environment active):
```
python scripts\audit_dataset.py
```
This read-only tool reports the number of videos, classes, CSV columns, resolution, FPS, durations, missing files,
and whether signer information exists. **Never modify the dataset files.**

If the automatic column detection is wrong, set `CSV_FILE`, `VIDEO_COLUMN`, `LABEL_COLUMN` (and `SIGNER_COLUMN`)
in `config.json`.
