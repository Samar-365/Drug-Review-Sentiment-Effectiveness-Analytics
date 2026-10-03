import os
import sys
import time
import tracemalloc


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


from ml_pipeline.models import (
    load_pipeline,
    predict_single_review,
)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "logistic_pipeline.joblib",
)

MEMORY_LIMIT_MB = 500

TEST_REVIEW = (
    "This medicine worked very well "
    "and improved my condition."
)


def bytes_to_mb(value):
    return value / (
        1024 * 1024
    )


def main():
    if not os.path.exists(
        MODEL_PATH
    ):
        print(
            "ERROR: Saved model pipeline "
            "was not found."
        )

        print(
            f"Expected path: {MODEL_PATH}"
        )

        sys.exit(1)

    print(
        "=" * 60
    )

    print(
        "SINGLE-REVIEW INFERENCE MEMORY PROFILE"
    )

    print(
        "=" * 60
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print()

    print(
        "Loading pipeline..."
    )

    load_start = (
        time.perf_counter()
    )

    pipeline = load_pipeline(
        MODEL_PATH
    )

    load_time = (
        time.perf_counter()
        - load_start
    )

    print(
        f"Pipeline load time: "
        f"{load_time:.4f} seconds"
    )

    print()

    tracemalloc.start()

    try:
        inference_start = (
            time.perf_counter()
        )

        result = (
            predict_single_review(
                TEST_REVIEW,
                pipeline,
            )
        )

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        current_memory, peak_memory = (
            tracemalloc.get_traced_memory()
        )

    finally:
        tracemalloc.stop()

    current_memory_mb = bytes_to_mb(
        current_memory
    )

    peak_memory_mb = bytes_to_mb(
        peak_memory
    )

    print(
        "Prediction:"
    )

    print(
        f"  Review: {TEST_REVIEW}"
    )

    print(
        f"  Sentiment: "
        f"{result['sentiment']}"
    )

    print(
        f"  Class: "
        f"{result['prediction']}"
    )

    if (
        result["confidence"]
        is not None
    ):
        print(
            f"  Confidence: "
            f"{result['confidence'] * 100:.2f}%"
        )

    if (
        result["probabilities"]
        is not None
    ):
        print(
            f"  Negative probability: "
            f"{result['probabilities']['negative']:.6f}"
        )

        print(
            f"  Positive probability: "
            f"{result['probabilities']['positive']:.6f}"
        )

    print()

    print(
        "Performance:"
    )

    print(
        f"  Inference time: "
        f"{inference_time:.6f} seconds"
    )

    print(
        f"  Current traced memory: "
        f"{current_memory_mb:.4f} MB"
    )

    print(
        f"  Peak traced memory: "
        f"{peak_memory_mb:.4f} MB"
    )

    print()

    if (
        peak_memory_mb
        < MEMORY_LIMIT_MB
    ):
        memory_status = "PASS"
    else:
        memory_status = "FAIL"

    print(
        f"Memory requirement "
        f"(< {MEMORY_LIMIT_MB} MB): "
        f"{memory_status}"
    )

    print(
        "=" * 60
    )

    if memory_status == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
    