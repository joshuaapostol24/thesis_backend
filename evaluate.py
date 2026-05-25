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

    conn = get_connection()

    cursor = conn.cursor(
        cursor_factory=RealDictCursor
    )

    cursor.execute("""
        SELECT
            risk_level,
            final_risk
        FROM risk_assessments
        ORDER BY timestamp DESC
        LIMIT 100
    """)

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    if not rows:
        print("No data found in risk_assessments table.")
        return

    # ---------------------------------------------
    # Risk Score Classification
    # ---------------------------------------------

    def to_label(score):

        if score >= 2.7:
            return "VERY HIGH"

        elif score >= 2.1:
            return "HIGH"

        elif score >= 1.2:
            return "MODERATE"

        elif score >= 0.5:
            return "LOW"

        else:
            return "VERY LOW"

    # ---------------------------------------------
    # Expected vs Predicted
    # ---------------------------------------------

    y_true = [
        row["risk_level"]
        for row in rows
    ]

    y_pred = [
        to_label(row["final_risk"])
        for row in rows
    ]

    # ---------------------------------------------
    # Accuracy Computation
    # ---------------------------------------------

    correct = sum([
        1
        for t, p in zip(y_true, y_pred)
        if t == p
    ])

    total = len(y_true)

    accuracy = correct / total

    print("\n=== ACCURACY EVALUATION ===\n")

    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {total - correct}")
    print(f"Total Predictions: {total}")

    print(
        f"Accuracy: "
        f"{accuracy:.2f} "
        f"({accuracy * 100:.1f}%)"
    )


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
