"""
modules/cnn_lstm.py
───────────────────
CNN + LSTM model for hazard risk prediction — per-barangay version.

Supports 15 barangays. Each barangay has its own:
  - Trained model weights (weights/cnn_lstm_barangay_XX.pt)
  - Training data loaded from PostgreSQL (barangay_training_data table)

Architecture
────────────
  CNN branch  : 1-D convolutions over point features (rainfall, flood)
  LSTM branch : processes SEQ_LEN historical readings
  Fusion head : CNN + LSTM → dense → scalar risk score in [0.0, 3.0]
"""

from __future__ import annotations

import logging
import os
import threading

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from modules.config import get_database_url

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

POINT_FEATURES = ["rainfall", "humidity", "soil", "flood", "storm_surge"]

INDICATOR_RANGES = {
    "rainfall":    (0.0, 50.0),
    "humidity":    (0.0, 100.0),
    "soil":        (0.0, 3.0),
    "flood":       (0.0, 4.0),
    "storm_surge": (0.0, 5.0),
}

BARANGAY_IDS: List[int] = list(range(1, 16))   # 1 – 15

_TRAIN_EPOCHS = 100   # Increased from 50 (longer sequences need more epochs)
_TRAIN_LR     = 1e-3
_BATCH_SIZE   = 128   # Reduced from 256 (90-day sequences use more GPU memory)
_MAX_ROWS     = 730
SEQ_LEN       = 90    # Increased from 10: Use full 90 days of historical data per requirement

# Paths
_WEIGHTS_DIR = Path(os.environ.get("BARANGAY_WEIGHTS_DIR", "weights"))
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def _weights_path(barangay_id: int) -> Path:
    return _WEIGHTS_DIR / f"cnn_lstm_barangay_{barangay_id:02d}.pt"


# ── Model definition ───────────────────────────────────────────────────────────

class CnnLstmRiskModel(nn.Module):
    def __init__(self, n_features=5, cnn_channels=32, lstm_hidden=64, fc_hidden=64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=1),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        fused = (cnn_channels * 2) + lstm_hidden

        self.fusion = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, point, seq):
        x_cnn = self.cnn(point.unsqueeze(-1)).mean(dim=-1)
        _, (h_n, _) = self.lstm(seq)
        fused = torch.cat([x_cnn, h_n[-1]], dim=1)
        return torch.clamp(self.fusion(fused).squeeze(-1), 0.0, 3.0)


# ── Normalisation helpers ──────────────────────────────────────────────────────

def _norm(value: float, key: str) -> float:
    lo, hi = INDICATOR_RANGES.get(key, (0.0, 1.0))
    return 0.0 if hi == lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _extract_point(E: dict):
    return [_norm(float(E.get(k, 0)), k) for k in POINT_FEATURES]


def _build_seq(H: dict, n_features: int):
    raw = []
    if isinstance(H.get("history"), list):
        for step in H["history"]:
            raw.append([_norm(float(step.get(k, 0)), k) for k in POINT_FEATURES])
    while len(raw) < SEQ_LEN:
        raw.insert(0, [0.0] * n_features)
    return raw[-SEQ_LEN:]


# ── Database data loader ───────────────────────────────────────────────────────

def load_barangay_data(barangay_id: int):
    """Load training data from PostgreSQL database."""
    import psycopg2

    conn = psycopg2.connect(get_database_url())
    cur = conn.cursor()

    cur.execute("""
        SELECT rainfall, humidity, soil, flood, storm_surge, risk_label
        FROM barangay_training_data
        WHERE barangay_id = %s
        ORDER BY id DESC
        LIMIT %s
    """, (barangay_id, _MAX_ROWS))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        raise ValueError(f"No training data found for barangay_id={barangay_id}")

    rows = list(reversed(rows))

    df = pd.DataFrame(
        rows,
        columns=["rainfall", "humidity", "soil", "flood", "storm_surge", "risk_label"]
    )

    df["r_n"]  = df["rainfall"].apply(lambda v: _norm(v, "rainfall"))
    df["s_n"]  = df["soil"].apply(lambda v: _norm(v, "soil"))
    df["f_n"]  = df["flood"].apply(lambda v: _norm(v, "flood"))
    df["h_n"]  = df["humidity"].apply(lambda v: _norm(v, "humidity"))
    df["ss_n"] = df["storm_surge"].apply(lambda v: _norm(v, "storm_surge"))

    points, seqs, labels = [], [], []

    for i in range(SEQ_LEN, len(df)):
        window = df.iloc[i - SEQ_LEN:i]
        seq = window[["r_n", "h_n", "s_n", "f_n", "ss_n"]].values.tolist()
        row = df.iloc[i]
        pt  = [row["r_n"], row["h_n"], row["s_n"], row["f_n"], row["ss_n"]]
        lbl = max(0.0, min(3.0, float(row["risk_label"])))
        points.append(pt)
        seqs.append(seq)
        labels.append(lbl)

    logger.info("Barangay %02d — loaded %d training samples from DB.", barangay_id, len(points))

    return (
        torch.tensor(points, dtype=torch.float32),
        torch.tensor(seqs,   dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )


# ── Training ───────────────────────────────────────────────────────────────────

def _train_model(barangay_id: int) -> CnnLstmRiskModel:
    logger.info("Training model for Barangay %02d …", barangay_id)

    pt_t, seq_t, lbl_t = load_barangay_data(barangay_id)
    loader = DataLoader(
        TensorDataset(pt_t, seq_t, lbl_t),
        batch_size=_BATCH_SIZE,
        shuffle=True
    )

    model     = CnnLstmRiskModel(n_features=len(POINT_FEATURES))
    optimizer = optim.Adam(model.parameters(), lr=_TRAIN_LR)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, _TRAIN_EPOCHS + 1):
        total = 0.0
        for pb, sb, lb in loader:
            optimizer.zero_grad()
            pred = model(pb, sb)
            loss = criterion(pred, lb)
            loss.backward()
            optimizer.step()
            total += loss.item() * pb.size(0)
        if epoch % 10 == 0:
            logger.info(
                "  Barangay %02d | Epoch %d/%d — MSE %.6f",
                barangay_id, epoch, _TRAIN_EPOCHS, total / len(lbl_t)
            )

    logger.info("Barangay %02d training complete.", barangay_id)
    return model


# ── Per-barangay model registry ────────────────────────────────────────────────

_registry:      Dict[int, CnnLstmRiskModel] = {}
_registry_lock = threading.Lock()


def _load_or_train_barangay(barangay_id: int) -> CnnLstmRiskModel:
    model = CnnLstmRiskModel(n_features=len(POINT_FEATURES))
    wp    = _weights_path(barangay_id)

    if wp.exists():
        try:
            model.load_state_dict(torch.load(wp, map_location="cpu"))
            logger.info("Barangay %02d — weights loaded from '%s'.", barangay_id, wp)
        except Exception as exc:
            logger.warning(
                "Barangay %02d — could not load weights (%s). Retraining.", barangay_id, exc
            )
            model = _train_model(barangay_id)
            torch.save(model.state_dict(), wp)
    else:
        model = _train_model(barangay_id)
        torch.save(model.state_dict(), wp)
        logger.info("Barangay %02d — weights saved to '%s'.", barangay_id, wp)

    model.eval()
    return model


def get_model(barangay_id: int) -> CnnLstmRiskModel:
    """Return the trained model for *barangay_id*, loading/training if needed."""
    if barangay_id not in BARANGAY_IDS:
        raise ValueError(f"barangay_id must be 1–15, got {barangay_id}.")

    with _registry_lock:
        if barangay_id not in _registry:
            _registry[barangay_id] = _load_or_train_barangay(barangay_id)
        return _registry[barangay_id]


# ── Parallel preload at startup ────────────────────────────────────────────────

def preload_all_models() -> None:
    """Load existing weights; train only if weights do not exist."""
    logger.info("Loading barangay models...")
    for bid in BARANGAY_IDS:
        try:
            get_model(bid)
            logger.info("✅ Barangay %02d ready.", bid)
        except Exception as e:
            logger.error("❌ Barangay %02d failed: %s", bid, e)
    logger.info("All barangay models ready.")


# ── Public inference API ───────────────────────────────────────────────────────

def predict_risk(barangay_id: int, E: dict, H: dict) -> float:
    if E is None:
        E = {}
    if H is None:
        H = {}

    model = get_model(barangay_id)

    pt  = torch.tensor([_extract_point(E)],                      dtype=torch.float32)
    seq = torch.tensor([_build_seq(H, len(POINT_FEATURES))], dtype=torch.float32)

    with torch.no_grad():
        score = float(model(pt, seq).item())

    logger.debug("Barangay %02d | risk=%.4f | E=%s", barangay_id, score, E)
    return score


def predict_all_barangays(
    readings:  Dict[int, dict],
    histories: Optional[Dict[int, dict]] = None
) -> Dict[int, float]:
    if histories is None:
        histories = {}
    return {
        bid: predict_risk(bid, e, histories.get(bid, {}))
        for bid, e in readings.items()
    }


def retrain_barangay(barangay_id: int, force: bool = False) -> None:
    """Retrain the model for one barangay using latest DB data."""
    if force:
        with _registry_lock:
            _registry.pop(barangay_id, None)

    model = _train_model(barangay_id)
    torch.save(model.state_dict(), _weights_path(barangay_id))
    model.eval()
    with _registry_lock:
        _registry[barangay_id] = model
    logger.info("Barangay %02d retrained and cached.", barangay_id)