"""Controlled follow-up experiment (paper Sections 2.3-2.5 and 3.2-3.5).

For each dataset (digits, iris, wine) and each of 20 partition seeds (100-119), a
multinomial logistic regression is fitted once and then corrected with three
post-hoc methods on nested calibration subsets of 10%, 25%, 50% and 100% of the
reserved calibration pool:

* Sigmoid logits   - scikit-learn's one-vs-rest Platt scaling on decision scores
* Sigmoid log odds - the same Platt fitter applied to class-vs-rest log odds
* Temperature      - a single positive temperature fitted by calibration NLL

Outputs (written next to this script):
    followup_results.csv   960 evaluation rows (incl. matched original rows)
    followup_summary.csv   means and sample SDs for all 48 conditions
    paired_changes.csv     720 corrected-minus-original rows
    splits.json            fitting/test indices and nested calibration ordering
    predictions.json.gz    every test probability vector
    environment.json       package versions

Requires scikit-learn 1.8.0 (see requirements.txt); the private
``_SigmoidCalibration`` fitter is version pinned.
"""
from pathlib import Path
import gzip
import io
import json

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.calibration import CalibratedClassifierCV, _SigmoidCalibration
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT_FOLDER = Path(__file__).resolve().parent
CRLF = '\r\n'  # committed outputs use CRLF; writing it explicitly avoids OS-dependent line endings

DATASETS = [('Digits', load_digits), ('Iris', load_iris), ('Wine', load_wine)]
SEEDS = range(100, 120)
CALIBRATION_FRACTIONS = [.1, .25, .5, 1.0]
ECE_BIN_COUNTS = [5, 10, 20]
CALIBRATION_ORDER_SEED_OFFSET = 10000
LOG_ODDS_CLIP = 1e-12


def measure(labels, probabilities):
    """Accuracy, log loss, summed Brier score, mean confidence and 5/10/20-bin ECE."""
    predicted = probabilities.argmax(axis=1)
    correct = predicted == labels
    confidence = probabilities.max(axis=1)
    target = np.eye(probabilities.shape[1])[labels]
    result = {
        'accuracy': correct.mean(),
        'log_loss': log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1])),
        'brier': np.sum((probabilities - target) ** 2, axis=1).mean(),
        'mean_confidence': confidence.mean(),
    }
    for number_of_bins in ECE_BIN_COUNTS:
        error = 0.0
        groups = np.minimum((confidence * number_of_bins).astype(int), number_of_bins - 1)
        for bin_number in range(number_of_bins):
            selected = groups == bin_number
            if selected.any():
                error += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
        result['ece' + str(number_of_bins)] = error
    return result


def nested_calibration_order(pool, labels, classes, seed):
    """Round-robin class sampling gives nested, approximately stratified subsets."""
    generator = np.random.default_rng(seed + CALIBRATION_ORDER_SEED_OFFSET)
    class_lists = [list(generator.permutation(pool[labels[pool] == k])) for k in range(classes)]
    order = []
    while any(class_lists):
        for class_list in class_lists:
            if class_list:
                order.append(int(class_list.pop()))
    return order


def fit_temperature(calibration_logits, calibration_labels):
    """Optimize one positive temperature on calibration data only (log-temperature space)."""
    sample_size = len(calibration_labels)

    def temperature_objective(log_temperature):
        scaled = calibration_logits / np.exp(log_temperature)
        losses = logsumexp(scaled, axis=1) - scaled[np.arange(sample_size), calibration_labels]
        return losses.mean()

    optimization = minimize_scalar(temperature_objective, bounds=(-4, 4), method='bounded',
                                   options={'xatol': 1e-8})
    assert optimization.success
    return np.exp(optimization.x)


def to_log_odds(probabilities):
    clipped = np.clip(probabilities, LOG_ODDS_CLIP, 1 - LOG_ODDS_CLIP)
    return np.log(clipped / (1 - clipped))


def main():
    rows = []
    split_records = []
    prediction_records = []

    for name, loader in DATASETS:
        features, labels = loader(return_X_y=True)
        for seed in SEEDS:
            # 70/30 development/test split, then 5/7 fitting and 2/7 calibration pool.
            development, test = train_test_split(np.arange(len(labels)), test_size=.3,
                                                 stratify=labels, random_state=seed)
            train, pool = train_test_split(development, test_size=2 / 7,
                                           stratify=labels[development], random_state=seed)
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
            model.fit(features[train], labels[train])
            classes = len(model.classes_)
            test_logits = model.decision_function(features[test])
            original = model.predict_proba(features[test])
            np.testing.assert_allclose(original, softmax(test_logits, axis=1), atol=1e-12)

            order = nested_calibration_order(pool, labels, classes, seed)
            split_records.append({'dataset': name, 'seed': seed, 'train': train.tolist(),
                                  'test': test.tolist(), 'calibration_order': order})

            for fraction in CALIBRATION_FRACTIONS:
                sample_size = max(2 * classes, int(round(len(pool) * fraction)))
                calibration = np.asarray(order[:sample_size])
                calibration_logits = model.decision_function(features[calibration])
                calibration_labels = labels[calibration]

                # Sigmoid on logits: the public API with one frozen prediction fold.
                all_calibration_indices = np.arange(sample_size)
                prediction_fold = [(all_calibration_indices, all_calibration_indices)]
                sigmoid = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid',
                                                 cv=prediction_fold, ensemble=False)
                sigmoid.fit(features[calibration], calibration_labels)
                sigmoid_probabilities = sigmoid.predict_proba(features[test])

                temperature = fit_temperature(calibration_logits, calibration_labels)
                temperature_probabilities = softmax(test_logits / temperature, axis=1)
                assert np.array_equal(temperature_probabilities.argmax(axis=1), original.argmax(axis=1))

                # Same version-pinned Platt fitter, but on class-versus-rest log odds.
                calibration_logodds = to_log_odds(model.predict_proba(features[calibration]))
                test_logodds = to_log_odds(original)
                raw_logodds = np.zeros_like(original)
                raw_logits = np.zeros_like(original)
                for k in range(classes):
                    binary_labels = (calibration_labels == k).astype(int)
                    fitted = _SigmoidCalibration().fit(calibration_logodds[:, k], binary_labels)
                    raw_logodds[:, k] = fitted.predict(test_logodds[:, k])
                    # Check the private fitter reproduces the public API on logits.
                    check = _SigmoidCalibration().fit(calibration_logits[:, k], binary_labels)
                    raw_logits[:, k] = check.predict(test_logits[:, k])
                np.testing.assert_allclose(raw_logits / raw_logits.sum(axis=1, keepdims=True),
                                           sigmoid_probabilities, atol=1e-10)
                logodds_probabilities = raw_logodds / raw_logodds.sum(axis=1, keepdims=True)

                variants = {
                    'Original': original,
                    'Sigmoid logits': sigmoid_probabilities,
                    'Temperature': temperature_probabilities,
                    'Sigmoid log odds': logodds_probabilities,
                }
                for method, probabilities in variants.items():
                    assert np.isfinite(probabilities).all()
                    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
                    rows.append({
                        'dataset': name, 'seed': seed, 'fraction': fraction,
                        'n_fit': len(train), 'n_calibration': sample_size, 'n_test': len(test),
                        'method': method,
                        'temperature': temperature if method == 'Temperature' else np.nan,
                        'label_change_rate': (probabilities.argmax(axis=1) != original.argmax(axis=1)).mean(),
                        **measure(labels[test], probabilities),
                    })
                    prediction_records.append({
                        'dataset': name, 'seed': seed, 'fraction': fraction, 'method': method,
                        'test_indices': test.tolist(), 'labels': labels[test].tolist(),
                        'probabilities': probabilities.tolist(),
                    })
            print(name, seed, flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_FOLDER / 'followup_results.csv', index=False, lineterminator=CRLF)
    summary_metrics = ['accuracy', 'log_loss', 'brier', 'ece10', 'mean_confidence', 'label_change_rate']
    (results.groupby(['dataset', 'fraction', 'n_calibration', 'method'])[summary_metrics]
     .agg(['mean', 'std'])
     .to_csv(OUTPUT_FOLDER / 'followup_summary.csv', lineterminator=CRLF))

    # Corrected-minus-original changes, matched on dataset, seed and calibration fraction.
    baseline = results[results.method == 'Original'].set_index(['dataset', 'seed', 'fraction'])
    changes = results[results.method != 'Original'].copy()
    for metric in ['accuracy', 'log_loss', 'brier', 'ece5', 'ece10', 'ece20', 'mean_confidence']:
        changes['delta_' + metric] = [row[metric] - baseline.loc[(row.dataset, row.seed, row.fraction), metric]
                                      for _, row in changes.iterrows()]
    changes.to_csv(OUTPUT_FOLDER / 'paired_changes.csv', index=False, lineterminator=CRLF)

    (OUTPUT_FOLDER / 'splits.json').write_text(json.dumps(split_records))
    # mtime=0 keeps the compressed file byte-identical across reruns.
    with open(OUTPUT_FOLDER / 'predictions.json.gz', 'wb') as raw, \
            gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed, \
            io.TextIOWrapper(compressed, encoding='utf-8') as stream:
        json.dump(prediction_records, stream)
    environment = {'sklearn': sklearn.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
                   'pandas': pd.__version__, 'seeds': [100, 119]}
    (OUTPUT_FOLDER / 'environment.json').write_bytes(json.dumps(environment, indent=2).replace('\n', CRLF).encode())
    print('Completed', len(results), 'evaluation rows')


if __name__ == '__main__':
    main()
