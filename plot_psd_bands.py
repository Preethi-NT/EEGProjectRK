import os
import math
import numpy as np
import pandas as pd
import mne

# -----------------------
# CONFIG
# -----------------------
INFO_CSV = "data/subject-info.csv"
EDF_DIR  = "data/eeg_files_2"          # only _2.edf files should be here
WINDOW_SEC = 5                          # <-- set your k (must divide 60)
FS_TARGET = 256                         # resample target (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 80),
}
PSD_FMIN, PSD_FMAX = 0.5, 45           # PSD band for total power
FOCUS_METRIC = "theta_beta_ratio"       # "theta_beta_ratio" | "beta_alpha_ratio" | "beta_rel_power"

# -----------------------
# HELPERS
# -----------------------
def welch_bandpower_window(window_chxT, fs, bands):
    """
    Compute Welch PSD per channel for a window (channels x samples).
    Returns dict of band -> average power across channels (float).
    """
    # mne's psd expects (n_channels, n_times)
    psd, freqs = mne.time_frequency.psd_array_welch(
        window_chxT,
        sfreq=fs,
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        n_fft=None,
        n_overlap=0,
        n_per_seg=min(window_chxT.shape[1], int(2*fs)),  # up to 2s segment
        average='mean',
        verbose=False
    )  # psd shape: (n_channels, n_freqs)

    band_powers = {}
    for name, (lo, hi) in bands.items():
        idx = (freqs >= lo) & (freqs <= hi)
        # mean power in band, averaged across freqs then across channels
        band_powers[name] = psd[:, idx].mean(axis=1).mean()
    # total power for relative calculations
    band_powers["total"] = psd.mean()
    return band_powers

def focus_score_from_bands(band_powers, metric="theta_beta_ratio"):
    """
    Convert band powers into a single concentration score for a window.
    - theta_beta_ratio: theta / beta  (lower -> higher focus)
    - beta_alpha_ratio: beta / alpha  (higher -> higher focus)
    - beta_rel_power:   beta / total  (higher -> higher focus)
    Returns float.
    """
    theta = band_powers["theta"]
    beta  = band_powers["beta"]
    alpha = band_powers["alpha"]
    total = max(band_powers.get("total", theta+beta+alpha+1e-12), 1e-12)

    if metric == "theta_beta_ratio":
        return float(theta / max(beta, 1e-12))
    if metric == "beta_alpha_ratio":
        return float(beta / max(alpha, 1e-12))
    if metric == "beta_rel_power":
        return float(beta / total)
    # default
    return float(theta / max(beta, 1e-12))

def windows_for_data(data_chxT, fs, window_sec):
    """
    Slice non-overlapping windows of window_sec from channels x samples array.
    Returns list of windows (each channels x samples), count = floor(T/window_sec).
    """
    samples_per_win = int(window_sec * fs)
    total_samples = data_chxT.shape[1]
    n_windows = total_samples // samples_per_win
    wins = []
    for w in range(n_windows):
        s = w * samples_per_win
        e = s + samples_per_win
        wins.append(data_chxT[:, s:e])
    return wins

def read_task_signal(path, fs_target):
    """
    Load EDF, keep EEG, resample, bandlimit to analysis range.
    Returns channels x samples numpy array and effective fs.
    """
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    # resample
    if fs_target is not None:
        raw.resample(fs_target)
    fs = raw.info["sfreq"]
    # gentle bandpass to PSD range (optional but helps)
    raw.filter(PSD_FMIN, PSD_FMAX, fir_design="firwin", verbose=False)
    return raw.get_data(), fs

def subject_id_from_filename(filename):
    """
    Assumes filenames like 'Subject00_2.edf' -> 'Subject00'
    Adjust if your naming differs.
    """
    base = os.path.basename(filename)
    # split at underscore, drop suffix
    return base.split("_")[0]

# -----------------------
# MAIN
# -----------------------
def main():
    # basic checks
    if 60 % WINDOW_SEC != 0:
        raise ValueError("WINDOW_SEC must divide 60 (e.g., 5, 6, 10, 12, 15, 20, 30).")

    # load subject-info
    df = pd.read_csv(INFO_CSV)

    # prepare columns Window1..WindowN
    n_cols = 60 // WINDOW_SEC
    win_cols = [f"Window{i+1}" for i in range(n_cols)]
    for col in win_cols:
        if col not in df.columns:
            df[col] = np.nan

    # index column for subject match (assumes a column named 'subject')
    if "subject" not in df.columns:
        raise ValueError("subject-info.csv must contain a 'subject' column to match filenames.")

    # process each _2.edf file
    files = [f for f in os.listdir(EDF_DIR) if f.lower().endswith("_2.edf")]
    files.sort()

    for fname in files:
        subj = subject_id_from_filename(fname)
        path = os.path.join(EDF_DIR, fname)

        try:
            data, fs = read_task_signal(path, FS_TARGET)
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")
            continue

        # make windows (non-overlapping)
        wins = windows_for_data(data, fs, WINDOW_SEC)

        # compute score per window
        scores = []
        for w in wins:
            bands = welch_bandpower_window(w, fs, BANDS)
            score = focus_score_from_bands(bands, metric=FOCUS_METRIC)
            scores.append(score)

        # pad/truncate to exactly n_cols
        if len(scores) < n_cols:
            scores = scores + [np.nan] * (n_cols - len(scores))
        elif len(scores) > n_cols:
            scores = scores[:n_cols]

        # write back to the row for this subject
        mask = (df["subject"].astype(str) == str(subj))
        if mask.any():
            df.loc[mask, win_cols] = [scores]
            print(f"[OK] {subj}: wrote {len(scores)} windows into subject-info.csv")
        else:
            print(f"[WARN] Subject id '{subj}' not found in subject-info.csv (skipping write)")

    # save
    df.to_csv(INFO_CSV, index=False)
    print(f"\nSaved updated file: {INFO_CSV}")
    print(f"Metric used: {FOCUS_METRIC} | Window length: {WINDOW_SEC}s | Columns: {n_cols}")

if __name__ == "__main__":
    main()
