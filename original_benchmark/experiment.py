"""Preliminary benchmark (paper Sections 2.1 and 3.1, Appendix B).

Two datasets (breast cancer, digits) x ten partition seeds x three classifiers
(logistic regression, random forest, Gaussian naive Bayes), each evaluated
before and after sigmoid calibration on a held-out calibration set.

Outputs (written next to this script):
    results.csv            120 evaluation rows
    reliability_bins.csv   674 occupied 10-bin reliability records
    summary.csv            means and sample SDs per dataset/model/variant
    environment.json       package versions
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_FOLDER = Path(__file__).resolve().parent
CRLF = '\r\n'  # committed outputs use CRLF; fixing it keeps reruns byte-identical on every OS
DATASETS = [('Breast cancer', load_breast_cancer), ('Digits', load_digits)]
NUMBER_OF_SEEDS = 10
NUMBER_OF_BINS = 10

RESULT_COLUMNS = ['dataset', 'seed', 'model', 'variant', 'n_fit', 'n_calibration', 'n_test',
                  'accuracy', 'log_loss', 'brier', 'ece', 'mean_confidence']
BIN_COLUMNS = ['dataset', 'seed', 'model', 'variant', 'bin', 'count', 'confidence', 'accuracy']


def make_models(seed):
    return {
        'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        'Random forest': RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        'Naive Bayes': GaussianNB(),
    }


def main():
    rows, bins = [], []
    for dataset, loader in DATASETS:
        X, y = loader(return_X_y=True)
        for seed in range(NUMBER_OF_SEEDS):
            # Keep the test set separate from both fitting and calibration:
            # 30% test, then 2/7 of the remaining 70% (= 20%) for calibration.
            X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=seed)
            X_fit, X_cal, y_fit, y_cal = train_test_split(X_dev, y_dev, test_size=2 / 7, stratify=y_dev,
                                                          random_state=seed)
            for name, model in make_models(seed).items():
                model.fit(X_fit, y_fit)
                calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid').fit(X_cal, y_cal)
                for variant, fitted in [('Original', model), ('Sigmoid', calibrated)]:
                    p = fitted.predict_proba(X_test)
                    predicted = fitted.classes_[p.argmax(axis=1)]
                    confidence, correct = p.max(axis=1), predicted == y_test
                    one_hot = (y_test[:, None] == fitted.classes_[None, :]).astype(float)

                    # Ten equal-width confidence bins; the last bin includes 1.0.
                    ece = 0.
                    for b in range(NUMBER_OF_BINS):
                        upper = confidence < (b + 1) / NUMBER_OF_BINS if b < NUMBER_OF_BINS - 1 else confidence <= 1
                        mask = (confidence >= b / NUMBER_OF_BINS) & upper
                        if mask.any():
                            bin_confidence, bin_accuracy = confidence[mask].mean(), correct[mask].mean()
                            ece += mask.mean() * abs(bin_confidence - bin_accuracy)
                            bins.append([dataset, seed, name, variant, b, int(mask.sum()),
                                         bin_confidence, bin_accuracy])

                    rows.append([dataset, seed, name, variant, len(y_fit), len(y_cal), len(y_test),
                                 accuracy_score(y_test, predicted), log_loss(y_test, p),
                                 np.mean(np.sum((p - one_hot) ** 2, axis=1)), ece, confidence.mean()])

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results.to_csv(OUTPUT_FOLDER / 'results.csv', index=False, lineterminator=CRLF)
    pd.DataFrame(bins, columns=BIN_COLUMNS).to_csv(OUTPUT_FOLDER / 'reliability_bins.csv', index=False,
                                                    lineterminator=CRLF)
    summary = results.groupby(['dataset', 'model', 'variant'])[['accuracy', 'log_loss', 'brier', 'ece']].agg(['mean', 'std'])
    summary.to_csv(OUTPUT_FOLDER / 'summary.csv', lineterminator=CRLF)
    environment = {'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}
    (OUTPUT_FOLDER / 'environment.json').write_bytes(json.dumps(environment, indent=2).replace('\n', CRLF).encode())
    print(summary.to_string())


if __name__ == '__main__':
    main()
