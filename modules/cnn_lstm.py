"""
modules/cnn_lstm.py
───────────────────
Adaptive CNN + LSTM hazard prediction system.

Per-barangay model architecture using:
    - Rainfall
    - Humidity
    - Soil saturation
    - Flood exposure
    - Storm surge exposure

Fully synchronized with:
    - weather_service.py
    - normalization.py
    - context.py
"""

from __future__ import annotations

import logging
import os
import threading

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import psycopg2
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from modules.config import get_database_url
from modules.normalization import (
    get_indicator_bounds,
    normalize,
)

logger = logging.getLogger(__name__)

# ── Device ────────────────────────────────────────────────────────────────────

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logger.info("CNN-LSTM device: %s", DEVICE)

# ── Feature configuration ─────────────────────────────────────────────────────

POINT_FEATURES = [
    "rainfall",
    "humidity",
    "soil",
    "flood",
    "storm_surge",
]

BARANGAY_IDS: List[int] = list(range(1, 16))

# ── Training configuration ────────────────────────────────────────────────────

_TRAIN_EPOCHS = 100
_TRAIN_LR     = 1e-3
_BATCH_SIZE   = 64
_MAX_ROWS     = 2000
SEQ_LEN       = 90

# ── Weights storage ───────────────────────────────────────────────────────────

_DEFAULT_WEIGHTS_DIR = (
    Path(__file__).resolve().parent.parent /
    "weights"
)

_WEIGHTS_DIR = Path(
    os.environ.get(
        "BARANGAY_WEIGHTS_DIR",
        str(_DEFAULT_WEIGHTS_DIR)
    )
)

_WEIGHTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def _weights_path(barangay_id: int) -> Path:
    return (
        _WEIGHTS_DIR /
        f"cnn_lstm_barangay_{barangay_id:02d}.pt"
    )


# ── Adaptive normalization ────────────────────────────────────────────────────

def _norm(
    value: float,
    key: str,
    barangay_id: int
) -> float:

    min_val, max_val = get_indicator_bounds(
        key,
        barangay_id
    )

    return normalize(
        value,
        min_val,
        max_val
    )


def _extract_point(
    E: dict,
    barangay_id: int
):

    return [
        _norm(
            float(E.get(feature, 0.0)),
            feature,
            barangay_id
        )
        for feature in POINT_FEATURES
    ]


def _build_seq(
    H: dict,
    barangay_id: int,
    n_features: int
):

    raw = []

    if isinstance(H.get("history"), list):

        for step in H["history"]:

            raw.append([
                _norm(
                    float(step.get(feature, 0.0)),
                    feature,
                    barangay_id
                )
                for feature in POINT_FEATURES
            ])

    while len(raw) < SEQ_LEN:
        raw.insert(0, [0.0] * n_features)

    return raw[-SEQ_LEN:]


# ── CNN + LSTM model ──────────────────────────────────────────────────────────

class CnnLstmRiskModel(nn.Module):

    def __init__(
        self,
        n_features=5,
        cnn_channels=32,
        lstm_hidden=64,
        fc_hidden=64
    ):
        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv1d(
                n_features,
                cnn_channels,
                kernel_size=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                cnn_channels,
                cnn_channels * 2,
                kernel_size=1
            ),

            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        fused_size = (
            (cnn_channels * 2) +
            lstm_hidden
        )

        self.fusion = nn.Sequential(

            nn.LayerNorm(fused_size),

            nn.Linear(
                fused_size,
                fc_hidden
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                fc_hidden,
                1
            ),
        )

    def forward(self, point, seq):

        cnn_out = self.cnn(
            point.unsqueeze(-1)
        ).mean(dim=-1)

        _, (hidden, _) = self.lstm(seq)

        fused = torch.cat(
            [cnn_out, hidden[-1]],
            dim=1
        )

        output = self.fusion(fused)

        return torch.clamp(
            output.squeeze(-1),
            0.0,
            3.0
        )


# ── Database loader ───────────────────────────────────────────────────────────

def load_barangay_data(
    barangay_id: int
):

    conn = psycopg2.connect(
        get_database_url()
    )

    cur = conn.cursor()

    cur.execute("""
        SELECT
            rainfall,
            humidity,
            soil,
            flood,
            storm_surge,
            risk_label
        FROM barangay_training_data
        WHERE barangay_id = %s
        ORDER BY id DESC
        LIMIT %s
    """, (
        barangay_id,
        _MAX_ROWS
    ))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    if not rows:

        raise ValueError(
            f"No training data for barangay_id={barangay_id}"
        )

    rows = list(reversed(rows))

    df = pd.DataFrame(
        rows,
        columns=[
            "rainfall",
            "humidity",
            "soil",
            "flood",
            "storm_surge",
            "risk_label",
        ]
    )

    # ── Adaptive normalization ────────────────────────────────────────

    for feature in POINT_FEATURES:

        df[f"{feature}_n"] = df[feature].apply(
            lambda value: _norm(
                value,
                feature,
                barangay_id
            )
        )

    points = []
    seqs   = []
    labels = []

    for i in range(SEQ_LEN, len(df)):

        window = df.iloc[i - SEQ_LEN:i]

        seq = window[
            [f"{f}_n" for f in POINT_FEATURES]
        ].values.tolist()

        row = df.iloc[i]

        point = [
            row[f"{f}_n"]
            for f in POINT_FEATURES
        ]

        label = float(
            max(
                0.0,
                min(3.0, row["risk_label"])
            )
        )

        points.append(point)
        seqs.append(seq)
        labels.append(label)

    logger.info(
        "Barangay %02d | loaded %d samples",
        barangay_id,
        len(points)
    )

    return (
        torch.tensor(points, dtype=torch.float32),
        torch.tensor(seqs, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def _train_model(
    barangay_id: int
) -> CnnLstmRiskModel:

    logger.info(
        "Training Barangay %02d model...",
        barangay_id
    )

    point_tensor, seq_tensor, label_tensor = (
        load_barangay_data(barangay_id)
    )

    dataset = TensorDataset(
        point_tensor,
        seq_tensor,
        label_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=_BATCH_SIZE,
        shuffle=True
    )

    model = CnnLstmRiskModel(
        n_features=len(POINT_FEATURES)
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=_TRAIN_LR
    )

    criterion = nn.MSELoss()

    model.train()

    for epoch in range(
        1,
        _TRAIN_EPOCHS + 1
    ):

        total_loss = 0.0

        for point_batch, seq_batch, label_batch in loader:

            point_batch = point_batch.to(DEVICE)
            seq_batch   = seq_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(
                point_batch,
                seq_batch
            )

            loss = criterion(
                predictions,
                label_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += (
                loss.item() *
                point_batch.size(0)
            )

        if epoch % 10 == 0:

            logger.info(
                "Barangay %02d | Epoch %d/%d | "
                "MSE %.6f",
                barangay_id,
                epoch,
                _TRAIN_EPOCHS,
                total_loss / len(label_tensor)
            )

    logger.info(
        "Barangay %02d training complete.",
        barangay_id
    )

    return model


# ── Registry ──────────────────────────────────────────────────────────────────

_registry: Dict[int, CnnLstmRiskModel] = {}

_registry_lock = threading.Lock()


def _load_or_train_barangay(
    barangay_id: int
) -> CnnLstmRiskModel:

    model = CnnLstmRiskModel(
        n_features=len(POINT_FEATURES)
    ).to(DEVICE)

    weights_path = _weights_path(
        barangay_id
    )

    if weights_path.exists():

        try:

            model.load_state_dict(
                torch.load(
                    weights_path,
                    map_location=DEVICE
                )
            )

            logger.info(
                "Barangay %02d | weights loaded",
                barangay_id
            )

        except Exception as exc:

            logger.warning(
                "Barangay %02d | "
                "weights corrupted (%s). "
                "Retraining...",
                barangay_id,
                exc
            )

            model = _train_model(
                barangay_id
            )

            torch.save(
                model.state_dict(),
                weights_path
            )

    else:

        model = _train_model(
            barangay_id
        )

        torch.save(
            model.state_dict(),
            weights_path
        )

        logger.info(
            "Barangay %02d | weights saved",
            barangay_id
        )

    model.eval()

    return model


def get_model(
    barangay_id: int
) -> CnnLstmRiskModel:

    if barangay_id not in BARANGAY_IDS:

        raise ValueError(
            f"barangay_id must be 1-15 "
            f"(got {barangay_id})"
        )

    with _registry_lock:

        if barangay_id not in _registry:

            _registry[barangay_id] = (
                _load_or_train_barangay(
                    barangay_id
                )
            )

        return _registry[barangay_id]


# ── Startup preload ───────────────────────────────────────────────────────────

def preload_all_models() -> None:

    logger.info(
        "Loading barangay models..."
    )

    for barangay_id in BARANGAY_IDS:

        try:

            get_model(barangay_id)

            logger.info(
                "✅ Barangay %02d ready",
                barangay_id
            )

        except Exception as exc:

            logger.error(
                "❌ Barangay %02d failed: %s",
                barangay_id,
                exc
            )

    logger.info(
        "All barangay models ready."
    )


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_risk(
    barangay_id: int,
    E: dict,
    H: dict
) -> float:

    E = E or {}
    H = H or {}

    model = get_model(
        barangay_id
    )

    point_tensor = torch.tensor(
        [_extract_point(E, barangay_id)],
        dtype=torch.float32
    ).to(DEVICE)

    seq_tensor = torch.tensor(
        [_build_seq(
            H,
            barangay_id,
            len(POINT_FEATURES)
        )],
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        score = float(
            model(
                point_tensor,
                seq_tensor
            ).item()
        )

    logger.debug(
        "Barangay %02d | "
        "predicted risk=%.4f",
        barangay_id,
        score
    )

    return score


def predict_all_barangays(
    readings: Dict[int, dict],
    histories: Optional[Dict[int, dict]] = None
) -> Dict[int, float]:

    histories = histories or {}

    return {
        barangay_id: predict_risk(
            barangay_id,
            reading,
            histories.get(barangay_id, {})
        )
        for barangay_id, reading
        in readings.items()
    }


def retrain_barangay(
    barangay_id: int,
    force: bool = False
) -> None:

    if force:

        with _registry_lock:
            _registry.pop(
                barangay_id,
                None
            )

    model = _train_model(
        barangay_id
    )

    torch.save(
        model.state_dict(),
        _weights_path(barangay_id)
    )

    model.eval()

    with _registry_lock:

        _registry[barangay_id] = model

    logger.info(
        "Barangay %02d retrained.",
        barangay_id
    )
