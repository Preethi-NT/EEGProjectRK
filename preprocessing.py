# Jai SiyaRam
import os
import numpy as np
import pandas as pd
import mne

# -----------------------
# CONFIG
# -----------------------
INFO_CSV = "data/subject-info.csv"
EDF_DIR = "data/eeg_files_2"        # only _2.edf files should be here
WINDOW_SEC = 6                      # must divide 60
FS_TARGET = 256                     # resample target (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
PSD_FMIN, PSD_FMAX = 0.5, 45
OUT_LONG = "data/window_features.csv"
OUT_WIDE = "data/ml_dataset_wide.csv"

# -----------------------
# HELPERS
# -----------------------
def _next_pow2(x: int) -> int:
    """Return next power of two >= x (x>0)."""
    if x <= 0:
        return 256
    return 1 << (int(np.ceil(np.log2(x))))

def welch_bandpower_window(window_chxT, fs, bands):
    """
    Compute PSD using Welch and integrate PSD over each band to obtain band power (µV^2).
    window_chxT: np.array shape (n_channels, n_samples)
    returns: dict with band_name -> power (float) averaged across channels, and 'total'.
    """
    win_len = window_chxT.shape[1]
    if win_len <= 0:
        # empty window
        empty = {k: 0.0 for k in bands.keys()}
        empty["total"] = 0.0
        return empty

    # --- parameters chosen for smooth PSD (more segments, less variance) ---
    # Use ~1 second segments for smoothing (fs samples)
    n_per_seg = min(win_len, int(fs))     # typically 256 samples (1s)
    n_overlap = n_per_seg // 2           # 50% overlap -> more averaging (smoother PSD)
    n_fft = max(_next_pow2(n_per_seg), 256)  # integer FFT length (power of two, >=256)

    # ensure valid integers
    n_per_seg = int(n_per_seg)
    n_overlap = int(min(n_overlap, n_per_seg - 1)) if n_per_seg > 1 else 0
    n_fft = int(n_fft)

    psd, freqs = mne.time_frequency.psd_array_welch(
        window_chxT,
        sfreq=fs,
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        average="mean",
        verbose=False,
    )
    # psd shape: (n_channels, n_freqs)

    # integration bin widths may be irregular in rare cases, use trapz with freqs
    band_powers = {}
    for name, (lo, hi) in bands.items():
        idx = (freqs >= lo) & (freqs <= hi)
        if idx.any():
            # integrate PSD over the band for each channel, then average channels
            power_per_channel = np.trapz(psd[:, idx], freqs[idx], axis=1)  # shape: (n_channels,)
            band_powers[name] = float(np.mean(power_per_channel))
        else:
            band_powers[name] = 0.0

    # total power integrated across PSD_FMIN..PSD_FMAX
    total_power_per_channel = np.trapz(psd, freqs, axis=1)
    band_powers["total"] = float(np.mean(total_power_per_channel))
    return band_powers

def windows_for_data(data_chxT, fs, window_sec):
    samples_per_win = int(window_sec * fs)
    total_samples = data_chxT.shape[1]
    n_windows = total_samples // samples_per_win
    return [data_chxT[:, w*samples_per_win:(w+1)*samples_per_win] for w in range(n_windows)]

def read_task_signal(path, fs_target):
    """
    Load EDF, pick EEG channels, resample to fs_target (if not None), and bandpass filter.
    Returns channels x samples array and effective sampling rate.
    """
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    # use new pick API to avoid legacy warning
    try:
        raw.pick(picks="eeg")
    except Exception:
        # fallback to pick_types if older MNE version
        raw.pick_types(eeg=True)
    if fs_target is not None:
        raw.resample(fs_target)
    fs = raw.info["sfreq"]
    raw.filter(PSD_FMIN, PSD_FMAX, fir_design="firwin", verbose=False)
    return raw.get_data(), fs

def subject_id_from_filename(filename):
    return os.path.basename(filename).split("_")[0]

# -----------------------
# MAIN
# -----------------------
def main():
    # basic check
    if 60 % WINDOW_SEC != 0:
        raise ValueError("WINDOW_SEC must divide 60 (e.g., 5, 6, 10, 12, 15, 20, 30).")

    # load subject-info
    df = pd.read_csv(INFO_CSV)

    # find subject column case-insensitively
    subj_cols = [c for c in df.columns if c.lower() == "subject"]
    if not subj_cols:
        raise ValueError("subject-info.csv must contain a 'subject' column (case-insensitive).")
    subj_col = subj_cols[0]

    # ensure Window columns exist
    n_cols = 60 // WINDOW_SEC
    win_cols = [f"Window{i+1}" for i in range(n_cols)]
    for col in win_cols:
        if col not in df.columns:
            df[col] = np.nan

    long_rows = []   # list of dicts: one per window
    wide_rows = []   # list of dicts: one per subject

    files = [f for f in os.listdir(EDF_DIR) if f.lower().endswith("_2.edf")]
    files.sort()

    if not files:
        print(f"[WARN] No _2.edf files found in {EDF_DIR}. Nothing to process.")
        return

    for fname in files:
        subj = subject_id_from_filename(fname)
        path = os.path.join(EDF_DIR, fname)
        print(f"[INFO] Processing file: {fname} (subject {subj})")

        try:
            data, fs = read_task_signal(path, FS_TARGET)
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")
            continue

        wins = windows_for_data(data, fs, WINDOW_SEC)
        if len(wins) == 0:
            print(f"[WARN] No windows extracted for {fname} (maybe non-60s file). Skipping.")
            continue

        scores = []
        wide_row = {"subject": subj}

        for i, w in enumerate(wins):
            bands = welch_bandpower_window(w, fs, BANDS)
            alpha_p = bands.get("alpha", 0.0)
            beta_p  = bands.get("beta", 0.0)
            theta_p = bands.get("theta", 0.0)
            total_p = bands.get("total", 1e-12)
            eps = 1e-12
            alpha_beta = alpha_p / max(beta_p, eps)
            theta_beta = theta_p / max(beta_p, eps)

            # add to long dataset (powers are integrated µV^2)
            long_rows.append({
                "subject": subj,
                "file": fname,
                "window_idx": i + 1,
                "alpha_power": alpha_p,
                "beta_power": beta_p,
                "theta_power": theta_p,
                "alpha_beta_ratio": alpha_beta,
                "theta_beta_ratio": theta_beta,
                "alpha_rel_power": alpha_p / max(total_p, eps),
                "beta_rel_power": beta_p / max(total_p, eps),
                "theta_rel_power": theta_p / max(total_p, eps),
            })

            # build wide format columns for this subject
            wide_row[f"Window{i+1}_alpha"] = alpha_p
            wide_row[f"Window{i+1}_beta"] = beta_p
            wide_row[f"Window{i+1}_theta"] = theta_p
            wide_row[f"Window{i+1}_alpha_beta"] = alpha_beta
            wide_row[f"Window{i+1}_theta_beta"] = theta_beta

            # scalar for subject-info Window columns (we store theta/beta ratio here)
            scores.append(theta_beta)

        # pad/truncate scores to exactly n_cols
        if len(scores) < n_cols:
            scores = scores + [np.nan] * (n_cols - len(scores))
        elif len(scores) > n_cols:
            scores = scores[:n_cols]

        mask = (df[subj_col].astype(str) == str(subj))
        if mask.any():
            arr = np.array(scores).reshape(1, -1)
            df.loc[mask, win_cols] = arr
            print(f"[OK] {subj}: wrote {len(scores)} Window columns into {INFO_CSV}")
        else:
            print(f"[WARN] Subject id '{subj}' not found in {INFO_CSV} (skipping write)")

        wide_rows.append(wide_row)

    # save outputs
    df.to_csv(INFO_CSV, index=False)
    pd.DataFrame(long_rows).to_csv(OUT_LONG, index=False)
    pd.DataFrame(wide_rows).to_csv(OUT_WIDE, index=False)

    print(f"\nSaved updated file: {INFO_CSV}")
    print(f"Saved long dataset (row = window): {OUT_LONG}  (rows = {len(long_rows)})")
    print(f"Saved wide dataset (row = subject): {OUT_WIDE} (rows = {len(wide_rows)})")
    print(f"Window length: {WINDOW_SEC}s | Expected Window columns: {n_cols}")

if __name__ == "__main__":
    main()
