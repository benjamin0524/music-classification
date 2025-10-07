# -*- coding: utf-8 -*-

import os, json, subprocess, tempfile, warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_ROOT   = "../hw1/artist20"
TRAIN_JSON  = os.path.join(DATA_ROOT, "train.json")
TEST_JSON   = os.path.join(DATA_ROOT, "test.json")
TEST_DIR    = os.path.join(DATA_ROOT, "test")

BEST_MODEL  = "best_model_cnn.pth"   
TEST_OUT    = "student_ID.json"

SR          = 16000
N_MELS      = 320
N_FFT       = 1024
HOP         = 160
SEG_LEN     = 30.0
NSEG_EVAL   = 9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = int(SR * SEG_LEN / HOP)

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
    return y

def get_duration_sec(path):
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return float(out)
    except Exception:
        y = ffmpeg_load(path, sr=SR)
        return len(y)/SR


def load_json_list(json_path):
    base_dir = os.path.dirname(os.path.abspath(json_path))
    with open(json_path, "r", encoding="utf-8") as f:
        rel_list = json.load(f)
    out = []
    for p in rel_list:
        p = os.path.normpath(p)
        out.append(p if os.path.isabs(p) else os.path.join(base_dir, p))
    return out

def extract_label_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    if "train_val" in parts:
        i = parts.index("train_val")
        if i + 1 < len(parts): return parts[i+1]
    return parts[-2]


def to_logmel(y, sr=SR):
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                       n_fft=N_FFT, hop_length=HOP, power=2.0)
    logS = librosa.power_to_db(S + 1e-10).astype(np.float32)
    logS = (logS - logS.mean()) / (logS.std() + 1e-8)  # per-sample z-score
    return logS

def pad_or_truncate(mel, max_len=MAX_LEN):
    if mel.shape[1] > max_len:  mel = mel[:, :max_len]
    elif mel.shape[1] < max_len: mel = np.pad(mel, ((0,0),(0, max_len - mel.shape[1])), mode="constant")
    return mel

def seg_offsets_eval(dur, seg_s=SEG_LEN, n=NSEG_EVAL):
    if dur <= seg_s: return [0.0]
    return list(np.linspace(0.0, dur - seg_s, n))


class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, c // r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, h, w = x.size()
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s

class Block(nn.Module):
    def __init__(self, c_in, c_out, stride=1, use_se=True, p_drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)
        self.se    = SEBlock(c_out) if use_se else nn.Identity()
        self.act   = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()
        self.proj  = nn.Conv2d(c_in, c_out, 1, stride, 0, bias=False) if (stride!=1 or c_in!=c_out) else None
    def forward(self, x):
        idn = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.drop(self.act(self.bn2(self.conv2(x))))
        x = self.se(x)
        if self.proj is not None: idn = self.proj(idn)
        return self.act(x + idn)

class StrongCNN(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(Block(64, 64), Block(64, 64))
        self.layer2 = nn.Sequential(Block(64, 128, stride=2), Block(128, 128, p_drop=0.1))
        self.layer3 = nn.Sequential(Block(128, 256, stride=2), Block(256, 256, p_drop=0.1))
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, n_class)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.head(x)


@torch.no_grad()
def predict_top3_for_file(model, fp, classes):
    dur = get_duration_sec(fp)
    probs = []
    for start in seg_offsets_eval(dur, SEG_LEN, NSEG_EVAL):
        y = ffmpeg_load(fp, sr=SR, offset=start, duration=SEG_LEN)
        mel = pad_or_truncate(to_logmel(y))
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
        out = model(x)
        probs.append(F.softmax(out, dim=1).cpu().numpy()[0])
    P = np.clip(np.array(probs), 1e-12, 1.0)
    gmean = np.exp(np.log(P).mean(axis=0))
    gmean = gmean / gmean.sum()
    idx = np.argsort(gmean)[::-1][:3]
    return [classes[i] for i in idx]

def main():
    train_files = load_json_list(TRAIN_JSON)
    classes = sorted(list(set(extract_label_from_path(f) for f in train_files)))
    n_class = len(classes)
    print(f"Classes: {n_class}")

    if not os.path.isfile(BEST_MODEL):
        raise FileNotFoundError(f"找不到模型權重：{BEST_MODEL}")
    model = StrongCNN(n_class).to(DEVICE)
    state = torch.load(BEST_MODEL, map_location=DEVICE)
    if isinstance(state, dict) and next(iter(state.values())).__class__.__name__.endswith("Tensor"):
        model.load_state_dict(state)
    else:
        model.load_state_dict(state.state_dict())
    model.eval()
    print(f"[LOAD] {BEST_MODEL}")

    # 3) 準備測試清單
    if os.path.isfile(TEST_JSON):
        test_files = load_json_list(TEST_JSON)
    elif os.path.isdir(TEST_DIR):
        test_files = [os.path.join(TEST_DIR, n)
                      for n in sorted(os.listdir(TEST_DIR))
                      if n.lower().endswith((".mp3", ".wav", ".flac"))]
    else:
        raise FileNotFoundError("未找到 test.json 或 test/ 目錄")

    # 4) 推論 & 輸出 JSON
    pred = {}
    for i, fp in enumerate(test_files, 1):
        try:
            top3 = predict_top3_for_file(model, fp, classes)
            key = os.path.splitext(os.path.basename(fp))[0]
            pred[key] = top3
        except Exception as e:
            print(f"[WARN] Skip {fp}: {e}")

        if i % 20 == 0:
            print(f"Inferred {i}/{len(test_files)} files...")

    with open(TEST_OUT, "w", encoding="utf-8") as f:
        json.dump(pred, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {TEST_OUT}  (total {len(pred)} items)")

if __name__ == "__main__":
    main()
