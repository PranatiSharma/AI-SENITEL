import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import psutil
import time
import os
import json
import argparse
from torch.utils.data import DataLoader, TensorDataset
# ============================================================
# Configuration
# ============================================================
WINDOW_SIZE     = 12     # 12 time-steps  (60 s at 5 s sampling)
INPUT_DIM       = 5      # cpu, mem, sent_rate, recv_rate, conns
HIDDEN_DIM      = 32
LATENT_DIM      = 8
NUM_LAYERS      = 1
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-3
EPOCHS          = 100
SAMPLE_INTERVAL = 5      # seconds
MODEL_PATH      = "lstm_autoencoder.pth"
CONFIG_PATH     = "detector_config.json"
BASELINE_FILE   = "baseline_capture.csv"
FEATURE_NAMES = [
    "cpu_percent", "memory_percent",
    "bytes_sent_rate", "bytes_recv_rate",
    "connections_count",
]
# ============================================================
# 1.  Helpers
# ============================================================
def normalise_window(window: np.ndarray) -> np.ndarray:
    """
    Per-feature zero-mean / unit-variance normalisation **within**
    the window.  If a feature is constant (std == 0) it becomes 0.
    Returns shape identical to input.
    """
    mean = window.mean(axis=0, keepdims=True)
    std  = window.std(axis=0, keepdims=True)
    std[std == 0] = 1.0          # avoid /0 for constant features
    return ((window - mean) / std).astype(np.float32)
def collect_one_sample(state: dict) -> list[float]:
    """
    Collect one sample of system metrics.
    `state` carries previous net counters for rate calculation.
    """
    cpu = psutil.cpu_percent(interval=0)
    mem = psutil.virtual_memory().percent
    net = psutil.net_io_counters()
    now = time.time()
    prev_net  = state.get("net")
    prev_time = state.get("time")
    if prev_net is None:
        state["net"]  = net
        state["time"] = now
        conns = float(len(psutil.net_connections()))
        return [cpu, mem, 0.0, 0.0, conns]
    dt = max(now - prev_time, 0.01)
    sent_rate = max(0.0, (net.bytes_sent - prev_net.bytes_sent) / dt)
    recv_rate = max(0.0, (net.bytes_recv - prev_net.bytes_recv) / dt)
    state["net"]  = net
    state["time"] = now
    conns = float(len(psutil.net_connections()))
    return [cpu, mem, sent_rate, recv_rate, conns]
# ============================================================
# 2.  LSTM Autoencoder
# ============================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True)
        self.enc2lat = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, num_layers,
                               batch_first=True)
        self.dec_out = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = self.enc2lat(h[-1])                              # (B, lat)
        z_rep = z.unsqueeze(1).repeat(1, x.size(1), 1)      # (B, S, lat)
        dec, _ = self.decoder(z_rep)
        return self.dec_out(dec)                              # (B, S, inp)
# ============================================================
# 3.  Baseline Collection
# ============================================================
def collect_baseline(out_file: str, duration_min: int = 15):
    state = {}
    n = (duration_min * 60) // SAMPLE_INTERVAL
    print(f"[COLLECT] Capturing {n} samples (~{duration_min} min) …")
    print("[COLLECT] Keep machine in NORMAL state.  Ctrl+C to stop early.\n")
    rows = []
    try:
        for i in range(n):
            rows.append(collect_one_sample(state))
            if (i + 1) % 12 == 0:
                print(f"  … {len(rows)} samples  "
                      f"({(i+1)*SAMPLE_INTERVAL}s elapsed)")
            time.sleep(SAMPLE_INTERVAL)
    except KeyboardInterrupt:
        print("\n[COLLECT] Stopped early.")
    if len(rows) < WINDOW_SIZE:
        print(f"[COLLECT] Need ≥{WINDOW_SIZE} samples, got {len(rows)}.")
        return
    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    df.to_csv(out_file, index=False)
    print(f"\n[COLLECT] Saved {len(df)} samples → {out_file}")
# ============================================================
# 4.  Trainer
# ============================================================
class Trainer:
    def __init__(self, data_file: str = None):
        self.data_file = data_file
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    # --- helpers -------------------------------------------------
    @staticmethod
    def _make_windows(raw: np.ndarray) -> np.ndarray:
        """Slide window over raw data, normalise each window internally."""
        wins = []
        for i in range(len(raw) - WINDOW_SIZE + 1):
            w = raw[i : i + WINDOW_SIZE]
            wins.append(normalise_window(w))
        return np.array(wins, dtype=np.float32)
    @staticmethod
    def _augment(windows: np.ndarray, factor: int = 30,
                 noise_std: float = 0.05) -> np.ndarray:
        """Create noisy copies so short baselines still train well."""
        parts = [windows]
        rng = np.random.default_rng(42)
        for _ in range(factor - 1):
            noise = rng.normal(0, noise_std, size=windows.shape
                               ).astype(np.float32)
            parts.append(windows + noise)
        return np.concatenate(parts)
    @staticmethod
    def _fallback_synthetic(n: int = 3000) -> np.ndarray:
        rng = np.random.default_rng(42)
        return np.column_stack([
            np.abs(rng.normal(30, 10, n)),
            np.clip(rng.normal(60, 3, n), 20, 95),
            np.abs(rng.exponential(120_000, n)),
            np.abs(rng.exponential(25_000, n)),
            np.abs(rng.poisson(200, n).astype(float)),
        ]).astype(np.float32)
    # --- main train ----------------------------------------------
    def train(self):
        # Load raw data
        if self.data_file and os.path.exists(self.data_file):
            print(f"[TRAIN] Loading {self.data_file}")
            raw = pd.read_csv(self.data_file).values.astype(np.float32)
        else:
            print("[TRAIN] No baseline — using synthetic data.")
            raw = self._fallback_synthetic()
        # Build self-normalised windows, then augment
        windows = self._make_windows(raw)
        aug     = self._augment(windows, factor=30, noise_std=0.05)
        np.random.default_rng(0).shuffle(aug)
        X = torch.from_numpy(aug).to(self.device)
        loader  = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE,
                             shuffle=True)
        # Model
        model = LSTMAutoencoder().to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=10, factor=0.5)
        crit  = nn.MSELoss()
        print(f"[TRAIN] Device={self.device}  Epochs={EPOCHS}  "
              f"Windows={len(aug)} (augmented from {len(windows)})")
        best_loss = float("inf")
        for ep in range(1, EPOCHS + 1):
            model.train()
            tot = 0.0
            for (bx,) in loader:
                optim.zero_grad()
                loss = crit(model(bx), bx)
                loss.backward()
                optim.step()
                tot += loss.item()
            avg = tot / len(loader)
            sched.step(avg)
            if avg < best_loss:
                best_loss = avg
                torch.save(model.state_dict(), MODEL_PATH)
            if ep % 10 == 0 or ep == 1:
                lr = optim.param_groups[0]["lr"]
                print(f"  Epoch {ep:3d}/{EPOCHS}  "
                      f"loss={avg:.6f}  best={best_loss:.6f}  lr={lr:.1e}")
        print(f"[TRAIN] Best model → {MODEL_PATH}")
        # --- Threshold on original (non-augmented) windows ---
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device,
                        weights_only=True))
        model.eval()
        orig = torch.from_numpy(windows).to(self.device)
        with torch.no_grad():
            recon = model(orig)
            errs = torch.mean((orig - recon) ** 2, dim=[1, 2]).cpu().numpy()
        mean_e = float(np.mean(errs))
        std_e  = float(np.std(errs))
        max_e  = float(np.max(errs))
        # Generous threshold: max training error × 2 safety margin
        threshold = max(max_e * 2.0, mean_e + 5 * std_e)
        cfg = {"threshold": threshold,
               "mean_train_error": mean_e,
               "std_train_error": std_e,
               "max_train_error": max_e}
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[TRAIN] Threshold = {threshold:.6f}  "
              f"(mean={mean_e:.6f}  std={std_e:.6f}  max={max_e:.6f})")
        print(f"[TRAIN] Config → {CONFIG_PATH}")
        return threshold
# ============================================================
# 5.  Real-time Detector
# ============================================================
class RealTimeDetector:
    CALIBRATION_STEPS = 24   # ~2 min warm-up
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"{CONFIG_PATH} missing — run --mode train")
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        self.threshold = cfg["threshold"]
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} missing — run --mode train")
        self.model = LSTMAutoencoder().to(self.device)
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device,
                        weights_only=True))
        self.model.eval()
        self.state  = {}      # for collect_one_sample
        self.buffer = []      # raw metric buffer
        self.cal_losses = []
    # ---- scoring ------------------------------------------------
    @staticmethod
    def _threat_score(loss: float, threshold: float) -> float:
        if threshold <= 0:
            return 0.0
        ratio = loss / threshold
        # Sigmoid centred at threshold (ratio=1 → score≈50)
        score = 100.0 / (1.0 + np.exp(-3.0 * (ratio - 1.0)))
        return float(np.clip(score, 0.0, 100.0))
    @staticmethod
    def _label(score: float) -> str:
        if score < 40:
            return "NORMAL"
        if score < 60:
            return "ELEVATED"
        if score < 80:
            return "BEHAVIORAL ANOMALY"
        return "CRITICAL ANOMALY"
    # ---- main loop -----------------------------------------------
    def monitor(self):
        print("=" * 65)
        print("  ANANTA — Real-time Behavioral Fingerprint Monitor")
        print("=" * 65)
        print(f"  Trained threshold : {self.threshold:.6f}")
        print(f"  Window            : {WINDOW_SIZE} × {SAMPLE_INTERVAL}s "
              f"= {WINDOW_SIZE * SAMPLE_INTERVAL}s")
        print(f"  Calibration       : first {self.CALIBRATION_STEPS} scores")
        print(f"  Device            : {self.device}")
        print("  Ctrl+C to stop.\n")
        # Warm up rate computation
        collect_one_sample(self.state)
        time.sleep(1)
        n_scores = 0
        try:
            while True:
                m = collect_one_sample(self.state)
                self.buffer.append(m)
                if len(self.buffer) > WINDOW_SIZE:
                    self.buffer.pop(0)
                if len(self.buffer) < WINDOW_SIZE:
                    left = WINDOW_SIZE - len(self.buffer)
                    print(f"  Filling buffer… {len(self.buffer)}/{WINDOW_SIZE} "
                          f"(~{left * SAMPLE_INTERVAL}s)")
                    time.sleep(SAMPLE_INTERVAL)
                    continue
                # --- Self-normalise the window ---
                window = np.array(self.buffer, dtype=np.float32)
                normed = normalise_window(window)
                x = torch.from_numpy(normed).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    recon = self.model(x)
                    loss  = torch.mean((x - recon) ** 2).item()
                n_scores += 1
                # --- Calibration phase ---
                if n_scores <= self.CALIBRATION_STEPS:
                    self.cal_losses.append(loss)
                    print(f"  [CALIBRATING {n_scores}/{self.CALIBRATION_STEPS}]  "
                          f"loss={loss:.6f}")
                    if n_scores == self.CALIBRATION_STEPS:
                        cal = np.array(self.cal_losses)
                        live_thresh = float(np.mean(cal) + 4 * np.std(cal))
                        self.threshold = max(self.threshold, live_thresh)
                        print(f"\n  [CALIBRATED]  threshold → {self.threshold:.6f}")
                        print(f"  (live mean={np.mean(cal):.6f}  "
                              f"std={np.std(cal):.6f})\n")
                    time.sleep(SAMPLE_INTERVAL)
                    continue
                # --- Scoring ---
                score = self._threat_score(loss, self.threshold)
                label = self._label(score)
                k = max(0, min(20, int(score / 5)))
                bar = "█" * k + "░" * (20 - k)
                print(f"  [{bar}]  Score: {score:5.1f}  "
                      f"Loss: {loss:.6f}  → {label}")
                time.sleep(SAMPLE_INTERVAL)
        except KeyboardInterrupt:
            print("\n\n  Monitoring stopped.")
# ============================================================
# 6.  CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="ANANTA — LSTM Behavioral Fingerprint Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Workflow:
  1) python lstm_anomaly_detector.py --mode collect --duration 15
  2) python lstm_anomaly_detector.py --mode train --file baseline_capture.csv
  3) python lstm_anomaly_detector.py --mode monitor
""")
    ap.add_argument("--mode", required=True,
                    choices=["collect", "train", "monitor"])
    ap.add_argument("--file", default=BASELINE_FILE)
    ap.add_argument("--duration", type=int, default=15)
    a = ap.parse_args()
    if a.mode == "collect":
        collect_baseline(a.file, a.duration)
    elif a.mode == "train":
        Trainer(data_file=a.file).train()
    elif a.mode == "monitor":
        RealTimeDetector().monitor()
if __name__ == "__main__":
    main()