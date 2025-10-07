# -*- coding: utf-8 -*-


import os, json, warnings, subprocess, tempfile, shutil
warnings.filterwarnings("ignore")

import numpy as np
import librosa
from tqdm import tqdm
from sklearn.preprocessing import normalize
import joblib

DATA_ROOT   = "../hw1/artist20"
TEST_DIR    = os.path.join(DATA_ROOT, "test")
OUTPUT_JSON = "student_ID.json"
MODEL_FILE  = "artist20_svm.pkl"
SCALER_FILE = "artist20_scaler.pkl"


SR = 16000
N_MELS = 128
N_MFCC = 40
N_FFT = 1024
HOP = 160
SEG_VAL = 12.0
NSEG_VAL = 7


def ffmpeg_load(path, sr=SR, offset=0.0, duration=None):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_name = tmp.name; tmp.close()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if offset and offset > 0: cmd += ["-ss", f"{float(offset):.6f}"]
    cmd += ["-i", path]
    if duration and duration > 0: cmd += ["-t", f"{float(duration):.6f}"]
    cmd += ["-ac", "1", "-ar", str(sr), tmp_name, "-y"]
    subprocess.run(cmd, check=True)
    y, _ = librosa.load(tmp_name, sr=sr, mono=True)
    os.remove(tmp_name)
    return y, sr

def get_duration_sec(path):
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


def seg_offsets_eval(dur, seg_s, n):
    if dur <= seg_s: return [0.0]
    return list(np.linspace(0.0, dur - seg_s, n))

def stats_2d(F):
    mu = F.mean(axis=1); sd = F.std(axis=1)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)

def extract_features(file_path, start_s, seg_s):
    y, sr = ffmpeg_load(file_path, sr=SR, offset=start_s, duration=seg_s)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP)
    logS = librosa.power_to_db(S + 1e-10)

    mfcc = librosa.feature.mfcc(S=logS, n_mfcc=N_MFCC)
    d1   = librosa.feature.delta(mfcc, order=1)
    d2   = librosa.feature.delta(mfcc, order=2)
    feat = [stats_2d(mfcc), stats_2d(d1), stats_2d(d2)]

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

    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness  = librosa.feature.spectral_flatness(y=y)[0]
    zcr       = librosa.feature.zero_crossing_rate(y=y)[0]
    rms       = librosa.feature.rms(y=y)[0]
    def ms(v): return np.array([v.mean(), v.std()], dtype=np.float32)
    feat += [ms(centroid), ms(bandwidth), ms(rolloff), ms(flatness), ms(zcr), ms(rms)]

    x = np.concatenate(feat, axis=0)
    x = np.nan_to_num(x)
    return x

def main():
    assert os.path.isfile(MODEL_FILE), f"❌ model not found: {MODEL_FILE}"
    assert os.path.isfile(SCALER_FILE), f"❌ scaler not found: {SCALER_FILE}"

    print(f"[INFO] Loading model: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    classes = list(model.classes_)

    test_files = sorted([
        os.path.join(TEST_DIR, f)
        for f in os.listdir(TEST_DIR)
        if f.lower().endswith(".mp3")
    ])
    print(f"[INFO] Found {len(test_files)} test files")

    results = {}
    for i, fp in enumerate(tqdm(test_files, desc="Infer test", ncols=100)):
        dur = get_duration_sec(fp)
        segs = seg_offsets_eval(dur, SEG_VAL, NSEG_VAL)
        feats = [extract_features(fp, s, SEG_VAL) for s in segs]
        feats = normalize(np.vstack(feats))
        feats_std = scaler.transform(feats)

        proba_seg = model.predict_proba(feats_std)
        logp = np.log(np.clip(proba_seg, 1e-12, 1.0))
        proba_avg = np.exp(logp.mean(axis=0))
        proba_avg /= proba_avg.sum()

        pred_idx = int(np.argmax(proba_avg))
        pred_label = classes[pred_idx]
        results[str(i+1)] = pred_label

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Done! Saved predictions → {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
