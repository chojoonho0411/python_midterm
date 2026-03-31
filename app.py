import argparse
import logging
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def parse_args():
    p = argparse.ArgumentParser(description="Train a simple ML model")
    p.add_argument("--test-size", type=float, default=0.2, help="Proportion for test split")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument("--n-estimators", type=int, default=100, help="RF n_estimators")
    p.add_argument("--output", type=str, default="model.joblib", help="Path to save model")
    return p.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def load_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

def build_model(n_estimators, random_state):
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

def main():
    args = parse_args()
    setup_logging()
    logging.info("Loading data")
    X, y = load_data()
    logging.info("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    logging.info("Building model")
    model = build_model(args.n_estimators, args.random_state)

    logging.info("Training")
    model.fit(X_train, y_train)

    logging.info("Evaluating")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logging.info("Accuracy: %.4f", acc)
    logging.info("Classification report:\n%s", classification_report(y_test, preds))

    out_path = Path(args.output)
    joblib.dump({"model": model}, out_path)
    logging.info("Saved model to %s", out_path.resolve())

if __name__ == "__main__":
    main()