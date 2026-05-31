"""
modules/simulation_notifier.py
──────────────────────────────
Sends FCM push notifications summarising a completed simulation run.

Requirements
------------
    pip install firebase-admin

Setup
-----
    1. Go to Firebase Console → Project Settings → Service Accounts
    2. Click "Generate new private key" → save as
       credentials/firebase_service_account.json
    3. Set env var (or keep the default path below):
           FIREBASE_CREDENTIALS=credentials/firebase_service_account.json
    4. Store FCM device tokens in your DB (one per client device).
       Pass them in via send_simulation_notification(tokens=[...]).
"""

from __future__ import annotations

import logging
import os

from typing import List

logger = logging.getLogger(__name__)

# ── Firebase initialisation (lazy, singleton) ────────────────────────────────

_firebase_initialised = False


def _init_firebase() -> None:
    global _firebase_initialised
    if _firebase_initialised:
        return

    import firebase_admin
    from firebase_admin import credentials

    cred_path = os.getenv(
        "FIREBASE_CREDENTIALS",
        "credentials/firebase_service_account.json",
    )

    if not os.path.exists(cred_path):
        raise FileNotFoundError(
            f"Firebase credentials not found at '{cred_path}'. "
            "Set the FIREBASE_CREDENTIALS env var or place the file at the default path."
        )

    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    _firebase_initialised = True
    logger.info("Firebase Admin SDK initialised.")


# ── Risk level helpers ───────────────────────────────────────────────────────

_RISK_EMOJI = {
    "VERY HIGH": "🔴",
    "HIGH":      "🟠",
    "MODERATE":  "🟡",
    "LOW":       "🟢",
    "VERY LOW":  "⚪",
}

_RISK_ORDER = {
    "VERY HIGH": 0,
    "HIGH":      1,
    "MODERATE":  2,
    "LOW":       3,
    "VERY LOW":  4,
}


def _build_notification_body(simulation_dict: dict) -> tuple[str, str]:
    """
    Returns (title, body) for the FCM notification.

    Title  — highest risk level found + barangay count
    Body   — bullet list of HIGH/VERY HIGH barangays, then summary counts
    """
    inputs    = simulation_dict.get("inputs", {})
    summary   = simulation_dict.get("summary", {})
    barangays = simulation_dict.get("barangays", [])

    # Highest risk level across all barangays
    sorted_b  = sorted(barangays, key=lambda b: _RISK_ORDER.get(b["risk_level"], 99))
    top_level = sorted_b[0]["risk_level"] if sorted_b else "UNKNOWN"
    top_emoji = _RISK_EMOJI.get(top_level, "❓")

    title = f"{top_emoji} Simulation Alert — {top_level} Risk Detected"

    # Weather line
    weather = (
        f"🌧 {inputs.get('rainfall', 0):.1f} mm/h  "
        f"💨 {inputs.get('wind_speed', 0):.1f} km/h  "
        f"💧 {inputs.get('humidity', 0):.1f}%  "
        f"🌡 {inputs.get('temperature', 0):.1f}°C"
    )

    # High-risk barangay list (VERY HIGH + HIGH only)
    high_risk = [
        b for b in barangays
        if b["risk_level"] in ("VERY HIGH", "HIGH")
    ]

    if high_risk:
        lines = [weather, "", "⚠️ High-risk barangays:"]
        for b in high_risk[:10]:          # cap at 10 to stay within FCM limits
            emoji = _RISK_EMOJI.get(b["risk_level"], "")
            lines.append(
                f"  {emoji} {b['barangay_name']} "
                f"(score: {b['final_score']:.2f})"
            )
    else:
        lines = [weather, "", "✅ No high-risk barangays detected."]

    # Summary footer
    lines += [
        "",
        f"Summary — "
        f"Very High: {summary.get('very_high', 0)}  "
        f"High: {summary.get('high', 0)}  "
        f"Moderate: {summary.get('moderate', 0)}  "
        f"Low: {summary.get('low', 0)}",
    ]

    body = "\n".join(lines)
    return title, body


# ── Public API ───────────────────────────────────────────────────────────────

def send_simulation_notification(
    simulation_dict: dict,
    tokens: List[str],
) -> dict:
    """
    Send an FCM push notification to one or more device tokens.

    Parameters
    ----------
    simulation_dict : dict
        The dict returned by SimulationResult.to_dict().
    tokens : list[str]
        FCM registration tokens for the target devices.

    Returns
    -------
    dict with keys:
        success_count   int
        failure_count   int
        failed_tokens   list[str]
        title           str
        body            str
    """
    if not tokens:
        logger.warning("send_simulation_notification called with no tokens.")
        return {
            "success_count": 0,
            "failure_count": 0,
            "failed_tokens": [],
            "title": "",
            "body":  "",
        }

    _init_firebase()

    from firebase_admin import messaging

    title, body = _build_notification_body(simulation_dict)

    message = messaging.MulticastMessage(
        tokens=tokens,
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        # Optional: add data payload so your app can navigate to the
        # simulation results screen on tap.
        data={
            "type":       "simulation_result",
            "risk_level": simulation_dict.get("barangays", [{}])[0].get(
                              "risk_level", "UNKNOWN"
                          ),
        },
        android=messaging.AndroidConfig(
            priority="high",
            notification=messaging.AndroidNotification(
                sound="default",
                channel_id="simulation_alerts",
            ),
        ),
        apns=messaging.APNSConfig(
            payload=messaging.APNSPayload(
                aps=messaging.Aps(sound="default"),
            ),
        ),
    )

    batch_response = messaging.send_each_for_multicast(message)

    failed_tokens = []
    for idx, resp in enumerate(batch_response.responses):
        if not resp.success:
            failed_tokens.append(tokens[idx])
            logger.warning(
                "FCM send failed for token %s: %s",
                tokens[idx][:20] + "…",
                resp.exception,
            )

    logger.info(
        "FCM notification sent | success=%d failure=%d",
        batch_response.success_count,
        batch_response.failure_count,
    )

    return {
        "success_count": batch_response.success_count,
        "failure_count": batch_response.failure_count,
        "failed_tokens": failed_tokens,
        "title":         title,
        "body":          body,
    }