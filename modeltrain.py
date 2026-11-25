import json  # noqa: I001
import math
import pickle
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
MODEL_DIR = ROOT / "saved_model"
MODEL_PATH = MODEL_DIR / "cataract_svm.pkl"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
IMAGE_SIZE = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_FOLDERS = (("normal", 0), ("cataract", 1))
RANDOM_STATE = 42
TEST_SIZE = 0.25


def extract_features(image_path: Path) -> np.ndarray:
    """Return the same HOG descriptor used by app.py for one image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("OpenCV could not decode this image")
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return hog(
        image,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True,
    )


def load_dataset():
    features, labels, skipped = [], [], []
    counts = {}
    for folder_name, label in CLASS_FOLDERS:
        folder = DATASET / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Required class folder is missing: {folder}")
        paths = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
        if not paths:
            raise ValueError(f"No supported images found in {folder}")
        successful = 0
        for path in paths:
            try:
                features.append(extract_features(path))
                labels.append(label)
                successful += 1
            except (ValueError, cv2.error) as exc:
                skipped.append((path.name, str(exc)))
        counts[folder_name] = successful
    if len(set(labels)) != 2:
        raise ValueError("Both class folders need at least one readable image.")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), counts, skipped


def print_dataset_summary(counts, skipped):
    total = sum(counts.values())
    print("\nDATASET SUMMARY")
    print("-" * 52)
    for name, count in counts.items():
        share = 100 * count / total if total else 0
        print(f"{name:12} {count:5} images ({share:5.1f}%)")
    print(f"{'total':12} {total:5} readable images")
    if skipped:
        print(f"\nSkipped {len(skipped)} unreadable file(s):")
        for filename, reason in skipped[:10]:
            print(f"  - {filename}: {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")


def main():
    X, y, counts, skipped = load_dataset()
    print_dataset_summary(counts, skipped)
    class_counts = Counter(y.tolist())
    smallest_class = min(class_counts.values())
    if smallest_class < 2:
        raise ValueError("Need at least 2 readable images in each class for a stratified split.")

    # Test size must be large enough to contain both classes after stratifying.
    test_count = max(2, math.ceil(len(y) * TEST_SIZE))
    if test_count >= len(y):
        raise ValueError("There are too few images for a train/test split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_count, random_state=RANDOM_STATE, stratify=y
    )

    # Scaling is fitted only on the training partition. class_weight helps
    # when one class has fewer images; it cannot fix incorrect labels/data.
    model = Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test, predictions, labels=[0, 1], target_names=["Normal", "Cataract"],
        zero_division=0, output_dict=True,
    )
    predicted_counts = Counter(int(value) for value in predictions)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "normal_recall": float(recall_score(y_test, predictions, pos_label=0, zero_division=0)),
        "cataract_recall": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "cataract_precision": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "confusion_matrix_labels": ["Normal", "Cataract"],
        "confusion_matrix": matrix.tolist(),
        "test_actual_counts": {"normal": int(np.sum(y_test == 0)), "cataract": int(np.sum(y_test == 1))},
        "test_predicted_counts": {"normal": predicted_counts.get(0, 0),
                                  "cataract": predicted_counts.get(1, 0)},
        "class_report": report,
        "dataset_counts": counts,
        "train_images": len(y_train),
        "test_images": len(y_test),
        "random_state": RANDOM_STATE,
        "test_fraction_requested": TEST_SIZE,
    }

    print("\nHELD-OUT TEST RESULTS")
    print("-" * 52)
    print(f"Accuracy:          {metrics['accuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Normal recall:     {metrics['normal_recall']:.3f}")
    print(f"Cataract recall:   {metrics['cataract_recall']:.3f}")
    print(f"Cataract precision:{metrics['cataract_precision']:.3f}")
    print(f"Macro F1:          {metrics['macro_f1']:.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted)")
    print("                    Pred normal   Pred cataract")
    print(f"Actual normal       {matrix[0,0]:11d}   {matrix[0,1]:13d}")
    print(f"Actual cataract     {matrix[1,0]:11d}   {matrix[1,1]:13d}")
    print("\nPer-class report")
    print(classification_report(y_test, predictions, labels=[0, 1],
                                target_names=["Normal", "Cataract"], zero_division=0))

    if len(predicted_counts) == 1:
        only = "cataract" if 1 in predicted_counts else "normal"
        print(f"WARNING: The held-out set was classified entirely as {only}.")
        print("Check the dataset labels, class counts, image variety, and camera/source differences.")
    if min(counts.values()) < 10:
        print("NOTE: Very few examples in at least one class; the hold-out metrics may vary a lot.")
    print("NOTE: Keep images from the same patient/source on only one side of the split when possible.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "class_names": {0: "Normal", 1: "Cataract"},
        "image_size": IMAGE_SIZE,
        "feature_method": "HOG",
        "feature_parameters": {"pixels_per_cell": (8, 8), "cells_per_block": (2, 2)},
        "metrics": metrics,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    with MODEL_PATH.open("wb") as file:
        pickle.dump(bundle, file, protocol=pickle.HIGHEST_PROTOCOL)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"\nSaved model bundle: {MODEL_PATH}")
    print(f"Saved evaluation metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
