"""
scripts/audit_dataset.py  --  READ-ONLY dataset inspector for ISL-CSLTR.

Run from the backend folder:
    python scripts/audit_dataset.py                (uses DATASET_PATH from the config)
    python scripts/audit_dataset.py "D:\\data\\ISL"   (or give a path)

It only READS files. It never changes, renames, deletes or creates anything
inside your dataset folder. The only file it writes is a text copy of the
report (reports/dataset_audit_report.txt in the project), outside the dataset.

Needs: pip install opencv-python pandas
"""

import os
import re
import sys
import statistics
from collections import Counter

import cv2
import pandas as pd

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".mpg", ".mpeg", ".m4v"}
SAMPLE_SIZE = 40        # how many videos to inspect in detail (all if fewer exist)
NEED_FRAMES = 30        # frames we plan to sample per video
NA = "Not available"


def folder_key(value):
    """Match sentence folders by case while ignoring whitespace only."""
    return re.sub(r"\s+", "", str(value)).casefold()


def csv_video_exists(reference, root, video_names, video_stems):
    """Resolve CSV-relative paths from the dataset root before basename fallback."""
    value = str(reference).strip().replace("\\", "/")
    if os.path.isfile(os.path.join(root, value)):
        return True
    base = os.path.basename(value).lower()
    return base in video_names or os.path.splitext(base)[0] in video_stems


# ---------- make output easy to copy: print to screen AND save to a text file ----------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def fmt(x, nd=2):
    return NA if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def read_csv_safely(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc), None
        except Exception as e:  # noqa
            last = e
    return None, str(last)


def main():
    # ---------- 1. dataset path ----------
    settings = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app.config import get_settings
        settings = get_settings()
    except Exception:
        pass

    if len(sys.argv) > 1:
        root = sys.argv[1]
    elif settings is not None:
        root = str(settings.dataset_dir)
        print(f"Using DATASET_PATH from config: {root}")
    else:
        root = input("Paste the ISL-CSLTR dataset folder path: ")
    root = root.strip().strip('"').strip("'")
    if not os.path.isdir(root):
        print(f"ERROR: '{root}' is not a folder. Check the path and try again.")
        return
    root = os.path.abspath(root)

    # set up report file (outside the dataset folder)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    report_dir = os.path.join(project_root, "reports")
    if os.path.abspath(report_dir).lower().startswith(root.lower()):
        report_dir = os.path.expanduser("~")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "dataset_audit_report.txt")
    report_file = open(report_path, "w", encoding="utf-8", errors="replace")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.stdout = Tee(sys.stdout, report_file)

    issues = []

    # ---------- 2. find files ----------
    videos, csvs = [], []
    for folder, _, files in os.walk(root):
        for f in files:
            full = os.path.join(folder, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO_EXTS:
                videos.append(full)
            elif ext == ".csv":
                csvs.append(full)
    videos.sort()
    csvs.sort()

    section("1. VIDEO FILES")
    print(f"Dataset folder : {root}")
    print(f"Total videos   : {len(videos)}")
    ext_count = Counter(os.path.splitext(v)[1].lower() for v in videos)
    print("Extensions     :", dict(ext_count) if ext_count else NA)

    per_folder = Counter(os.path.relpath(os.path.dirname(v), root) for v in videos)
    print("\nVideos per folder:")
    for k, n in sorted(per_folder.items()):
        print(f"  {n:6d}  {k}")

    names = Counter(os.path.basename(v).lower() for v in videos)
    folder_videos = {}
    folder_dirs = {}
    for video in videos:
        folder = os.path.dirname(video)
        key = folder_key(os.path.basename(folder))
        folder_videos.setdefault(key, []).append(video)
        folder_dirs.setdefault(key, set()).add(folder)
    ambiguous_folders = {key for key, paths in folder_dirs.items() if len(paths) > 1}
    dups = {k: n for k, n in names.items() if n > 1}
    if dups:
        issues.append(f"{len(dups)} video file names appear more than once (in different folders)")
    empty = [v for v in videos if os.path.getsize(v) == 0]
    if empty:
        issues.append(f"{len(empty)} video files are empty (0 bytes)")

    # ---------- 3. CSV files ----------
    section("2. CSV FILES")
    print(f"CSV files found: {len(csvs)}")
    video_names = {os.path.basename(v).lower() for v in videos}
    video_stems = {os.path.splitext(n)[0] for n in video_names}
    best = None  # (score, csv_path, df, video_col)
    folder_best = None  # (score, csv_path, df, label_col)
    csv_summary = NA
    csv_candidates = csvs
    if settings and settings.CSV_FILE:
        forced_csv = os.path.join(root, settings.CSV_FILE)
        if os.path.isfile(forced_csv):
            csv_candidates = [forced_csv]
        else:
            print(f"Configured CSV_FILE not found: {forced_csv}; auto-detecting instead.")
    for c in csv_candidates:
        df, err = read_csv_safely(c)
        print(f"\n--- {os.path.relpath(c, root)}")
        if df is None:
            print("  Could not read this CSV:", err)
            issues.append(f"CSV could not be read: {os.path.basename(c)}")
            continue
        print("  Columns:", list(df.columns))
        print("  Rows   :", len(df))
        print("  First 5 rows:")
        print(df.head(5).to_string(max_colwidth=60))
        # which column looks like video file names?
        for col in df.columns:
            vals = df[col].dropna().astype(str)
            if vals.empty:
                continue
            hits = sum(
                (os.path.basename(v.replace("\\", "/")).lower() in video_names)
                or (os.path.splitext(os.path.basename(v.replace("\\", "/")))[0].lower() in video_stems)
                for v in vals
            )
            score = hits / len(vals)
            if hits > 0 and (best is None or score > best[0]):
                best = (score, c, df, col)
            folder_col = settings.LABEL_COLUMN if settings and settings.LABEL_COLUMN in df.columns else col
            if col != folder_col:
                continue
            folder_hits = sum(
                folder_key(value) in folder_videos and folder_key(value) not in ambiguous_folders
                for value in vals
            )
            folder_score = folder_hits / len(vals)
            if folder_hits > 0 and (folder_best is None or folder_score > folder_best[0]):
                folder_best = (folder_score, c, df, col)

    # ---------- 4. label mapping ----------
    section("3. VIDEO-TO-LABEL MAPPING")
    unique_classes = NA
    missing_count = NA
    signer_info = NA
    if best is not None and (folder_best is None or best[0] >= folder_best[0]):
        _, cpath, df, vcol = best
        csv_summary = f"{os.path.basename(cpath)} ({len(df)} rows)"
        print(f"CSV used        : {os.path.relpath(cpath, root)}")
        print(f"Video column    : {vcol}")

        # guess the label column
        hints = ("sentence", "label", "text", "class", "gloss", "word", "caption", "transcript")
        other = [c for c in df.columns if c != vcol]
        label_col = next((c for c in other if any(h in str(c).lower() for h in hints)), None)
        if label_col is None:
            obj = [c for c in other if df[c].dtype == object]
            label_col = obj[0] if obj else None
        print(f"Label column    : {label_col if label_col else NA} (guessed from column names)")

        # missing references
        refs = df[vcol].dropna().astype(str)
        ref_keys = set()
        missing = []
        for r in refs:
            base = os.path.basename(r.replace("\\", "/")).lower()
            stem = os.path.splitext(base)[0]
            ref_keys.add(base)
            ref_keys.add(stem)
            if not csv_video_exists(r, root, video_names, video_stems):
                missing.append(r)
        missing_count = len(missing)
        unreferenced = [v for v in video_names if v not in ref_keys and os.path.splitext(v)[0] not in ref_keys]
        print(f"\nCSV rows pointing to a video that does NOT exist: {len(missing)}")
        for m in missing[:10]:
            print("   ", m)
        print(f"Video files NOT mentioned in the CSV: {len(unreferenced)}")
        if missing:
            issues.append(f"{len(missing)} CSV rows reference missing videos")
        if unreferenced:
            issues.append(f"{len(unreferenced)} videos are not listed in the CSV")

        if label_col is not None:
            # only count rows whose video file really exists
            def _exists(r):
                return csv_video_exists(r, root, video_names, video_stems)
            ok_rows = df[df[vcol].apply(_exists)]
            counts = ok_rows[label_col].astype(str).str.strip().value_counts()
            unique_classes = len(counts)
            print(f"\nUnique labels ({label_col}): {unique_classes}")
            print(f"Videos per label (existing videos only) -> min {counts.min()}, max {counts.max()}, mean {counts.mean():.2f}")
            print("How many labels have N videos:", dict(sorted(Counter(int(n) for n in counts.values).items())))
            print("Class distribution (videos per class, all classes):")
            print("  " + ", ".join(f"{int(n)}" for n in counts.values))
            print("5 smallest labels:")
            for lab, n in counts.tail(5).items():
                print(f"   {n:4d}  {str(lab)[:70]}")

        # signer information in CSV columns?
        sig_hints = ("signer", "subject", "person", "participant", "speaker", "user", "actor", "volunteer")
        sig_cols = [c for c in df.columns if any(h in str(c).lower() for h in sig_hints)]
        if sig_cols:
            parts = []
            for c in sig_cols:
                parts.append(f"column '{c}' has {df[c].nunique()} distinct values")
            signer_info = "; ".join(parts)
        else:
            signer_info = "No signer column found in CSV (" + NA + " from CSV)"
        print("\nSigner information:", signer_info)
        # signer hints in folder / file names
        name_hint = [v for v in videos[:2000] if "signer" in v.lower() or "person" in v.lower()]
        if name_hint:
            print(f"Note: {len(name_hint)}+ video paths contain 'signer'/'person' - check folder names.")
    elif folder_best is not None:
        _, cpath, df, label_col = folder_best
        csv_summary = f"{os.path.basename(cpath)} ({len(df)} rows)"
        print(f"CSV used        : {os.path.relpath(cpath, root)}")
        print("Video column    : Not present; videos are grouped by sentence folder")
        print(f"Label column    : {label_col} (matched to sentence folder names)")

        counts = Counter()
        missing_labels = []
        matched_keys = set()
        for value in df[label_col].dropna().astype(str):
            key = folder_key(value)
            matches = folder_videos.get(key, []) if key not in ambiguous_folders else []
            if not matches:
                missing_labels.append(value)
                continue
            counts[value.strip()] += len(matches)
            matched_keys.add(key)

        missing_count = len(missing_labels)
        print(f"\nCSV sentence labels with no unambiguous video folder: {missing_count}")
        for label in missing_labels[:10]:
            print("   ", label)
        unmatched_dirs = [key for key in folder_videos
                          if key not in matched_keys and key not in ambiguous_folders]
        unmatched_videos = sum(len(folder_videos[key]) for key in unmatched_dirs)
        print(f"Video files in folders not listed in the CSV: {unmatched_videos}")
        if missing_labels:
            issues.append(f"{len(missing_labels)} CSV sentence labels have no matching video folder")
        if unmatched_videos:
            issues.append(f"{unmatched_videos} videos are in folders not listed in the CSV")
        if ambiguous_folders:
            ambiguous_count = sum(len(folder_videos[key]) for key in ambiguous_folders)
            print(f"Video files in ambiguous sentence folders: {ambiguous_count}")
            issues.append(f"{len(ambiguous_folders)} sentence folder names are ambiguous")

        unique_classes = len(counts)
        if counts:
            values = list(counts.values())
            print(f"\nUnique labels ({label_col}): {unique_classes}")
            print(f"Videos per label -> min {min(values)}, max {max(values)}, mean {statistics.mean(values):.2f}")
            print("How many labels have N videos:", dict(sorted(Counter(values).items())))
            print("Class distribution (videos per class, all classes):")
            print("  " + ", ".join(str(n) for n in values))
            print("5 smallest labels:")
            for label, n in sorted(counts.items(), key=lambda item: item[1])[:5]:
                print(f"   {n:4d}  {label[:70]}")

        sig_hints = ("signer", "subject", "person", "participant", "speaker", "user", "actor", "volunteer")
        sig_cols = [c for c in df.columns if any(h in str(c).lower() for h in sig_hints)]
        if sig_cols:
            signer_info = "; ".join(f"column '{c}' has {df[c].nunique()} distinct values" for c in sig_cols)
        else:
            signer_info = "No signer column found in CSV (" + NA + " from CSV)"
        print("\nSigner information:", signer_info)
    else:
        print("No CSV column matched video filenames or sentence folders. Mapping: " + NA)
        issues.append("Could not match a CSV column to video filenames or sentence folders")

    # ---------- 5. inspect videos ----------
    section("4. VIDEO PROPERTIES")
    if not videos:
        print("No videos found - nothing to inspect.")
        sample = []
    elif len(videos) <= SAMPLE_SIZE:
        sample = videos
    else:
        step = len(videos) / SAMPLE_SIZE
        sample = [videos[int(i * step)] for i in range(SAMPLE_SIZE)]
    print(f"Inspecting {len(sample)} of {len(videos)} videos (evenly spread across the dataset)\n")

    rows = []
    for v in sample:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            print(f"  UNREADABLE: {os.path.relpath(v, root)}")
            issues.append(f"Unreadable video: {os.path.basename(v)}")
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip() or NA
        counted = 0
        while cap.grab():          # read-only: just steps through frames
            counted += 1
        cap.release()
        fps_ok = fps and fps > 0
        dur = counted / fps if fps_ok else None
        rows.append(dict(name=os.path.basename(v), fmt=os.path.splitext(v)[1].lower(), codec=codec,
                         w=w, h=h, fps=round(fps, 2) if fps_ok else None,
                         frames=counted, reported=reported, dur=dur))
        print(f"  {os.path.basename(v)[:38]:38s} {os.path.splitext(v)[1].lower():5s} "
              f"{w}x{h}  fps={fmt(rows[-1]['fps'])}  frames={counted}  dur={fmt(dur)}s  codec={codec}")

    def stats(vals):
        vals = [x for x in vals if x is not None]
        if not vals:
            return NA, NA, NA
        return min(vals), max(vals), statistics.mean(vals)

    section("5. STATISTICS (from inspected videos)")
    res_txt = fps_txt = dur_txt = frm_txt = fmt_txt = "Not available"
    thirty_txt = NA
    if rows:
        d_min, d_max, d_avg = stats([r["dur"] for r in rows])
        f_min, f_max, f_avg = stats([r["frames"] for r in rows])
        res = Counter(f"{r['w']}x{r['h']}" for r in rows)
        fpsd = Counter(r["fps"] for r in rows)
        fm = Counter(f"{r['fmt']} ({r['codec']})" for r in rows)
        print("Duration (s): min", fmt(d_min), "| max", fmt(d_max), "| avg", fmt(d_avg))
        print("Frames      : min", fmt(f_min), "| max", fmt(f_max), "| avg", fmt(f_avg))
        print("Resolution distribution:", dict(res))
        print("FPS distribution       :", dict(fpsd))
        print("Format distribution    :", dict(fm))
        mismatch = [r for r in rows if r["reported"] != r["frames"]]
        if mismatch:
            print(f"Note: {len(mismatch)} videos report a different frame count than actually decoded (decoded count is used).")

        enough = sum(1 for r in rows if r["frames"] >= NEED_FRAMES)
        short = [r for r in rows if r["frames"] < NEED_FRAMES]
        print(f"\nVideos with at least {NEED_FRAMES} frames: {enough} of {len(rows)}")
        if short:
            print(f"Videos with fewer than {NEED_FRAMES} frames:")
            for r in short[:10]:
                print(f"   {r['name']} ({r['frames']} frames)")
            issues.append(f"{len(short)} sampled videos have fewer than {NEED_FRAMES} frames")
        thirty_txt = f"{enough} of {len(rows)} sampled videos have >= {NEED_FRAMES} frames (shortest: {f_min})"

        res_txt = f"{dict(res)}"
        fps_txt = f"{dict(fpsd)}"
        dur_txt = f"min {fmt(d_min)}s, max {fmt(d_max)}s, avg {fmt(d_avg)}s"
        frm_txt = f"min {fmt(f_min)}, max {fmt(f_max)}, avg {fmt(f_avg)}"
        fmt_txt = f"{dict(fm)}"

    # ---------- summary ----------
    print("\n" + "#" * 70)
    print("DATASET AUDIT SUMMARY")
    print("#" * 70)
    print(f"Total videos: {len(videos)}")
    print(f"Unique classes: {unique_classes}")
    print(f"CSV: {csv_summary}")
    print(f"Video format: {fmt_txt}")
    print(f"Resolution: {res_txt}")
    print(f"FPS: {fps_txt}")
    print(f"Duration: {dur_txt}")
    print(f"Frame count: {frm_txt}")
    print(f"Signer information: {signer_info}")
    print(f"30-frame sampling: {thirty_txt}")
    print(f"Missing files: {missing_count}")
    print("Other issues:", "; ".join(issues) if issues else "None found")
    print("#" * 70)
    print(f"\n(A copy of this report was saved to: {report_path})")


if __name__ == "__main__":
    main()
