# -*- coding: utf-8 -*-
"""
Artist20 - Traditional ML (boosted accuracy with model saving)
- Segment sampling: train 5x12s / val&test 7x12s
- Features (302-D): MFCC(40)+Δ+ΔΔ mean/std + chroma + spectral_contrast + tonnetz + scalars(mean/std)
- L2 normalize -> StandardScaler
- SVM(RBF, probability=True, class_weight='balanced') with GridSearchCV
- Robust decoding via ffmpeg
- Saves: Top-1/Top-3 + confusion matrices + test Top-3 JSON
- NEW: save model & scaler as .pkl for inference
"""

import os, json, time, warnings, subprocess, shutil, tempfile
warnings.filterwarnings("ignore")

# 強制用 ffmpeg
os.environ.setdefault("AUDIOREAD_BACKEND", "ffmpeg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, top_k_accuracy_score
import joblib  # <-- 新增

# ==================== PATHS ====================
DATA_ROOT  = "/home/benjamin/music/hw1/artist20"
TRAIN_JSON = os.path.join(DATA_ROOT, "train.json")
VAL_JSON   = os.path.join(DATA_ROOT, "val.json")
TEST_DIR   = os.path.join(DATA_ROOT, "test")
TEST_JSON  = None
OUTPUT_TEST_JSON = "student_ID.json"
MODEL_FILE = "artist20_svm.pkl"
SCALER_FILE = "artist20_scaler.pkl"
# ==============================================

# ==================== AUDIO/FEAT ====================
SR = 16000
N_MELS = 128
N_MFCC = 40
N_FFT = 1024
HOP = 160
SEG_TRAIN = 8.0
SEG_VAL   = 8.0
NSEG_TRAIN = 20
NSEG_VAL   = 7
SEED = 42
np.random.seed(SEED)
# ====================================================

# ---------- ffmpeg helpers ----------
def have_ffmpeg():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

def ffmpeg_load(path, sr=SR, offset=0.0, duration=None):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_name = tmp.name; tmp.close()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if offset and offset > 0: cmd += ["-ss", f"{float(offset):.6f}"]
    cmd += ["-i", path]
    if duration and duration > 0: cmd += ["-t", f"{float(duration):.6f}"]
    cmd += ["-ac", "1", "-ar", str(sr), tmp_name, "-y"]
    subprocess.run(cmd, check=True)
    y, _ = librosa.load(tmp_name, sr=sr, mono=True)
    try: os.remove(tmp_name)
    except Exception: pass
    return y, sr

def safe_load(path, sr=SR, offset=0.0, duration=None):
    return ffmpeg_load(path, sr=sr, offset=offset, duration=duration)

def get_duration_sec(path):
    if have_ffmpeg():
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            val = float(out)
            if val > 0: return val
        except Exception:
            pass
    y, _ = ffmpeg_load(path, sr=SR)
    return len(y)/float(SR)

# ---------- path & labels ----------
def load_json_list(json_path):
    base_dir = os.path.dirname(os.path.abspath(json_path))
    with open(json_path, "r", encoding="utf-8") as f:
        rel_list = json.load(f)
    out = []
    for p in rel_list:
        p = os.path.normpath(p)
        if p.startswith("." + os.sep): p = p[2:]
        out.append(p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p)))
    return out

def extract_label_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    if "train_val" in parts:
        i = parts.index("train_val")
        if i + 1 < len(parts): return parts[i+1]
    return parts[-2]

# ---------- segmentation ----------
def seg_offsets_train(dur, seg_s, n):
    if dur <= seg_s: return [0.0]
    rng = np.random.RandomState(SEED + int(dur*1000) % 9973)
    starts = rng.uniform(0.0, max(1e-6, dur - seg_s), size=n)
    return list(np.sort(starts))

def seg_offsets_eval(dur, seg_s, n):
    if dur <= seg_s: return [0.0]
    return list(np.linspace(0.0, dur - seg_s, n))

# ---------- features ----------
def stats_2d(F):
    mu = F.mean(axis=1); sd = F.std(axis=1)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)

def extract_features(file_path, start_s, seg_s):
    y, sr = safe_load(file_path, sr=SR, offset=float(max(0.0, start_s)), duration=float(seg_s))
    if y.size == 0: raise RuntimeError("empty audio")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, power=2.0)
    logS = librosa.power_to_db(S + 1e-10)

    mfcc = librosa.feature.mfcc(S=logS, n_mfcc=N_MFCC)
    d1   = librosa.feature.delta(mfcc, order=1)
    d2   = librosa.feature.delta(mfcc, order=2)
    feat = [stats_2d(mfcc), stats_2d(d1), stats_2d(d2)]  # 240

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    feat.append(stats_2d(chroma))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)
    feat.append(stats_2d(contrast))
    try:
        y_h = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_h, sr=sr)
        feat.append(stats_2d(tonnetz))
    except Exception:
        feat.append(np.zeros(12, dtype=np.float32))

    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    flatness  = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP)[0]
    zcr       = librosa.feature.zero_crossing_rate(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    rms       = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    def ms(v): return np.array([v.mean(), v.std()], dtype=np.float32)
    feat += [ms(centroid), ms(bandwidth), ms(rolloff), ms(flatness), ms(zcr), ms(rms)]

    x = np.concatenate(feat, axis=0)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

# ---------- dataset builders ----------
def build_train_matrix(file_list, n_segments=NSEG_TRAIN, seg_s=SEG_TRAIN):
    X, y = [], []
    for fp in tqdm(file_list, desc="Extract train", ncols=100):
        if not os.path.isfile(fp): continue
        dur = get_duration_sec(fp)
        for off in seg_offsets_train(dur, seg_s, n_segments):
            X.append(extract_features(fp, off, seg_s))
            y.append(extract_label_from_path(fp))
    X = np.vstack(X); y = np.array(y)
    X = normalize(X)
    return X, y

def build_eval_tracks(file_list, n_segments=NSEG_VAL, seg_s=SEG_VAL):
    out = []
    for fp in tqdm(file_list, desc="Extract val/test", ncols=100):
        if not os.path.isfile(fp): continue
        dur = get_duration_sec(fp)
        feats = [extract_features(fp, off, seg_s) for off in seg_offsets_eval(dur, seg_s, n_segments)]
        feats = normalize(np.vstack(feats))
        out.append((feats, extract_label_from_path(fp), fp))
    return out

# ---------- main ----------
def main():
    t0 = time.time()
    train_files = load_json_list(TRAIN_JSON)
    val_files   = load_json_list(VAL_JSON)
    print(f"#train = {len(train_files)} | #val = {len(val_files)}")

    print("\n== 2) Extract TRAIN features ==")
    X_train, y_train = build_train_matrix(train_files)

    print("\n== 3) Scale & GridSearch SVM ==")
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)

    param_grid = {
        "C": [1, 2, 5, 10, 20],
        "gamma": ["scale", 1e-3, 5e-4, 1e-4]
    }
    base = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(base, param_grid, scoring="accuracy", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_train_std, y_train)
    model = gs.best_estimator_
    print("Best params:", gs.best_params_)

    # === Save model & scaler ===
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"[SAVED] model → {MODEL_FILE}, scaler → {SCALER_FILE}")

    print("\n== 4) Extract VAL features ==")
    val_tracks = build_eval_tracks(val_files)

    print("\n== 5) Evaluate ==")
    classes = list(np.unique(y_train))
    model_classes = list(model.classes_)
    reorder = [model_classes.index(c) for c in classes]

    y_true, y_pred, y_score = [], [], []
    for feats, lab, _ in tqdm(val_tracks, desc="Infer val", ncols=100):
        feats_std = scaler.transform(feats)
        proba_seg = model.predict_proba(feats_std)
        logp = np.log(np.clip(proba_seg, 1e-12, 1.0))
        proba_avg = np.exp(logp.mean(axis=0))[reorder]
        proba_avg = proba_avg / proba_avg.sum()
        y_score.append(proba_avg); y_true.append(lab)
        y_pred.append(classes[int(np.argmax(proba_avg))])

    y_true = np.array(y_true); y_pred = np.array(y_pred); y_score = np.vstack(y_score)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    top1 = top_k_accuracy_score(y_true, y_score, k=1, labels=classes)
    top3 = top_k_accuracy_score(y_true, y_score, k=3, labels=classes)
    print(f"Validation Top-1: {top1:.4f} | Top-3: {top3:.4f}\n")
    print("Classification report:")
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))

    for mat, title, fname in [
        (cm, "Validation Confusion Matrix", "val_confusion_matrix.png"),
        (cm.astype(float)/cm.sum(axis=1, keepdims=True), "Validation Confusion Matrix (Row-normalized)", "val_confusion_matrix_norm.png"),
    ]:
        plt.figure(figsize=(12,10))
        sns.heatmap(mat, cmap="Blues", annot=False, xticklabels=classes, yticklabels=classes)
        plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout(); plt.savefig(fname, dpi=220); plt.close()
        print(f"Saved: {fname}")

    print(f"\nAll done ✅  time used: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
