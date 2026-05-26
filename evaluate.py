import sys
import os
import time
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.database import get_connection
from psycopg2.extras import RealDictCursor



# =====================================================
# DEPLOYED BACKEND URL
# =====================================================

BASE_URL = "https://resq-app-xsb98.ondigitalocean.app/api"


# =====================================================
# 1. ACCURACY EVALUATION
# =====================================================

def evaluate_accuracy():

    import torch
    import numpy as np

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    conn = get_connection()

    cursor = conn.cursor(
        cursor_factory=RealDictCursor
    )

    # =============================================
    # USE barangay_training_data
    # =============================================

    cursor.execute("""
        SELECT
            barangay_id,
            risk_level,
            risk_label
        FROM barangay_training_data
        ORDER BY timestamp DESC
        LIMIT 1000
    """)

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    if not rows:

        print(
            "No data found in "
            "barangay_training_data table."
        )

        return

    # =============================================
    # LABEL ENCODING
    # =============================================

    label_map = {
        "VERY LOW": 0,
        "LOW": 1,
        "MODERATE": 2,
        "HIGH": 3,
        "VERY HIGH": 4,
    }

    # =============================================
    # TRUE LABELS
    # =============================================

    y_true = np.array([
        label_map[row["risk_level"]]
        for row in rows
    ])

    # =============================================
    # RULE-BASED EVALUATION
    # =============================================

    def to_label(score):

        if score >= 2.7:
            return 4

        elif score >= 2.1:
            return 3

        elif score >= 1.2:
            return 2

        elif score >= 0.5:
            return 1

        else:
            return 0

    rule_predictions = np.array([
        to_label(row["risk_label"])
        for row in rows
    ])

    rule_accuracy = accuracy_score(
        y_true,
        rule_predictions
    )

    print("\n=====================================")

    print(
        " RULE-BASED EVALUATION "
    )

    print("=====================================\n")

    print(
        f"Accuracy: "
        f"{rule_accuracy * 100:.2f}%"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            rule_predictions,
            labels=list(label_map.values()),
            target_names=list(label_map.keys()),
            zero_division=0
        )
    )

    print("\nConfusion Matrix:\n")

    print(
        confusion_matrix(
            y_true,
            rule_predictions,
            labels=list(label_map.values())
        )
    )

    # =============================================
    # CNN-LSTM EVALUATION
    # =============================================

    print("\n=====================================")

    print(
        " CNN-LSTM EVALUATION "
    )

    print("=====================================\n")

    from modules.cnn_lstm import (
        CnnLstmRiskModel,
        load_barangay_data,
    )

    weights_dir = "weights"

    model_files = [
        f for f in os.listdir(weights_dir)
        if f.endswith(".pt")
    ]

    if not model_files:

        print("No trained models found.")

        return

    for barangay_id, model_file in enumerate(model_files, start=1):

        print(
            f"\nEvaluating "
            f"{model_file}"
        )

        model_path = os.path.join(
            weights_dir,
            model_file
        )

        try:

            # LOAD REAL TRAINING DATA
            point_tensor, seq_tensor, label_tensor = (
                load_barangay_data(barangay_id)
            )

            # CREATE MODEL
            model = CnnLstmRiskModel()

            # LOAD WEIGHTS
            model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=torch.device("cpu")
                )
            )

            # EVAL MODE
            model.eval()

            with torch.no_grad():

                outputs = model(
                    point_tensor,
                    seq_tensor
                )

            # PREDICTIONS
            y_pred = torch.round(
                outputs
            ).numpy().astype(int)

            # LIMIT RANGE
            y_pred = np.clip(
                y_pred,
                0,
                4
            )

            # TRUE LABELS
            y_true_cnn = torch.round(
                label_tensor
            ).numpy().astype(int)

            accuracy = accuracy_score(
                y_true_cnn,
                y_pred
            )

            print(
                f"\nAccuracy: "
                f"{accuracy * 100:.2f}%"
            )

            print(
                "\nClassification Report:\n"
            )

            print(
                classification_report(
                    y_true_cnn,
                    y_pred,
                    labels=list(label_map.values()),
                    target_names=list(label_map.keys()),
                    zero_division=0
                )
            )

            print(
                "\nConfusion Matrix:\n"
            )

            print(
                confusion_matrix(
                    y_true_cnn,
                    y_pred,
                    labels=list(label_map.values())
                )
            )

        except Exception as e:

            print(
                f"\nFailed to evaluate "
                f"{model_file}"
            )

            print(e)
# =====================================================
# 2. SPEED EVALUATION
# =====================================================

def evaluate_speed():

    trials = []

    print("\n=== SPEED TEST ===\n")

    for i in range(5):

        start = time.time()

        response = requests.get(
            f"{BASE_URL}/health"
        )

        end = time.time()

        elapsed = end - start

        trials.append(elapsed)

        print(
            f"Trial {i+1}: "
            f"{elapsed:.4f} seconds"
        )

    average = sum(trials) / len(trials)

    print(
        f"\nAverage Response Time: "
        f"{average:.4f} seconds"
    )


# =====================================================
# 3. RELIABILITY EVALUATION
# =====================================================

def evaluate_reliability():

    success = 0
    fail = 0

    print("\n=== RELIABILITY TEST ===\n")

    for i in range(10):

        try:

            response = requests.get(
                f"{BASE_URL}/health"
            )

            if response.status_code == 200:

                success += 1

                print(
                    f"Trial {i+1}: SUCCESS"
                )

            else:

                fail += 1

                print(
                    f"Trial {i+1}: FAILED"
                )

        except Exception as e:

            fail += 1

            print(
                f"Trial {i+1}: ERROR"
            )

            print(e)

    reliability = (
        success / (success + fail)
    ) * 100

    print("\n=== RELIABILITY SUMMARY ===\n")

    print(f"Successful Requests: {success}")
    print(f"Failed Requests: {fail}")

    print(
        f"Reliability: "
        f"{reliability:.1f}%"
    )


# =====================================================
# RUN ALL TESTS
# =====================================================

if __name__ == "__main__":

    print(
        "\n====================================="
    )

    print(
        " RESQ SYSTEM PERFORMANCE EVALUATION "
    )

    print(
        "=====================================\n"
    )

    print("1. ACCURACY EVALUATION")
    evaluate_accuracy()

    print("\n2. SPEED EVALUATION")
    evaluate_speed()

    print("\n3. RELIABILITY EVALUATION")
    evaluate_reliability()

    print(
        "\n====================================="
    )

    print(
        " EVALUATION COMPLETED "
    )

    print(
        "=====================================\n"
    )
