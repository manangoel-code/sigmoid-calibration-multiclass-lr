"""Check the paper's printed numbers and the saved records against each other.

1. Every numeric cell of the paper's tables (extracted into
   verification/printed_tables.json) is compared with values recomputed from
   original_benchmark/*.csv, followup_results.csv and paired_changes.csv at the
   printed decimal precision.
2. All 960 saved probability vectors in predictions.json.gz are re-scored and
   must reproduce followup_results.csv to 1e-12.
3. Fitting/calibration/test indices in splits.json must be disjoint and
   complete, and temperature scaling must never change a predicted label.

Reports are written to verification/benchmark_verification.json and
verification/verification_report.json. The script exits with an error if any
printed value disagrees with the rerun.
"""
from pathlib import Path
import gzip
import json
import re

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parent
VERIFICATION = ROOT / 'verification'

MODELS = ['Logistic regression', 'Random forest', 'Naive Bayes']
BENCHMARK_DATASETS = ['Breast cancer', 'Digits']
FOLLOWUP_DATASETS = ['Digits', 'Iris', 'Wine']
FRACTIONS = [.1, .25, .5, 1.0]
METHODS = ['Original', 'Sigmoid logits', 'Sigmoid log odds', 'Temperature']
CORRECTIONS = METHODS[1:]

# Printed tables keyed by their asset id in the source PDF (see printed_tables.json).
TABLE_1, TABLE_2, TABLE_3 = 6, 8, 9
TABLE_B1_B2 = [(17, 'Breast cancer'), (18, 'Digits')]
TABLE_B3_B4 = [(19, ['log_loss', 'brier']), (20, ['ece', 'accuracy'])]
TABLE_B5 = 21
TABLE_B6 = [(22, 'Breast cancer'), (23, 'Breast cancer'), (24, 'Digits'), (25, 'Digits')]
TABLE_C_MAIN = [(26, 'Digits'), (28, 'Iris'), (30, 'Wine')]
TABLE_C_CONFIDENCE = [(27, 'Digits'), (29, 'Iris'), (31, 'Wine')]
TABLE_C_PAIRED = [(32, 'Digits'), (33, 'Iris'), (34, 'Wine')]
TABLE_C10 = 35
TABLE_D3 = 38

checks = []
mismatches = []


def check_value(label, cell, expected):
    """Compare every number in a printed cell with the expected value(s) at printed precision."""
    printed_numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', str(cell))
    expected = expected if isinstance(expected, (list, tuple)) else [expected]
    if len(printed_numbers) != len(expected):
        mismatches.append((label, cell, expected, 'parse'))
        return
    for text, want in zip(printed_numbers, expected):
        decimals = len(text.split('.')[1]) if '.' in text else 0
        tolerance = .5001 * 10 ** (-decimals) + 1e-10
        ok = abs(float(text) - float(want)) <= tolerance
        checks.append((label, text, float(want), ok))
        if not ok:
            mismatches.append((label, text, float(want), float(text) - float(want)))


def mean_sd(values, factor=1):
    return [values.mean() * factor, values.std(ddof=1) * factor]


def score(labels, probabilities):
    """Same metric definitions as investigate.measure, recomputed independently."""
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    values = {
        'accuracy': correct.mean(),
        'log_loss': log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1])),
        'brier': np.sum((probabilities - np.eye(probabilities.shape[1])[labels]) ** 2, axis=1).mean(),
        'mean_confidence': confidence.mean(),
    }
    for number_of_bins in [5, 10, 20]:
        group = np.minimum((confidence * number_of_bins).astype(int), number_of_bins - 1)
        ece = 0
        for bin_number in range(number_of_bins):
            mask = group == bin_number
            if mask.any():
                ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
        values['ece' + str(number_of_bins)] = ece
    return values


def verify_benchmark(printed, benchmark, bins):
    # Table 1: original vs sigmoid LL and ECE, and accuracy change.
    for i, row in enumerate(printed[TABLE_1][1:]):
        ds, model = BENCHMARK_DATASETS[i // 3], MODELS[i % 3]
        g = benchmark[(benchmark.dataset == ds) & (benchmark.model == model)]
        orig, cal = g[g.variant == 'Original'], g[g.variant == 'Sigmoid']
        expected = [orig.log_loss.mean(), cal.log_loss.mean(), orig.ece.mean(), cal.ece.mean(),
                    100 * (cal.accuracy.mean() - orig.accuracy.mean())]
        for j, x in enumerate(expected):
            check_value(f'Table1 {ds} {model} c{j}', row[j + 2], x)

    # Tables B1-B2: complete summaries.
    for aid, ds in TABLE_B1_B2:
        for i, row in enumerate(printed[aid][1:]):
            model, variant = MODELS[i // 2], ['Original', 'Sigmoid'][i % 2]
            g = benchmark[(benchmark.dataset == ds) & (benchmark.model == model) & (benchmark.variant == variant)]
            for j, (metric, factor) in enumerate([('accuracy', 100), ('log_loss', 1), ('brier', 1), ('ece', 1)]):
                check_value(f'Table B summary {ds} {model} {variant} {metric}', row[j + 2], mean_sd(g[metric], factor))

    # Tables B3-B4: paired changes and number of improving partitions.
    for aid, metrics in TABLE_B3_B4:
        for i, row in enumerate(printed[aid][1:]):
            ds, model = BENCHMARK_DATASETS[i // 3], MODELS[i % 3]
            g = benchmark[(benchmark.dataset == ds) & (benchmark.model == model)]
            original = g[g.variant == 'Original'].set_index('seed')
            cal = g[g.variant == 'Sigmoid'].set_index('seed')
            for j, metric in enumerate(metrics):
                delta = cal[metric] - original[metric]
                factor = 100 if metric == 'accuracy' else 1
                check_value(f'Paired benchmark {ds} {model} {metric}', row[2 + 2 * j], mean_sd(delta, factor))
                gains = int((delta > 0).sum() if metric == 'accuracy' else (delta < 0).sum())
                check_value(f'Paired gains {ds} {model} {metric}', row[3 + 2 * j], [gains, 10])

    # Table B5: mean confidence and accuracy.
    for i, row in enumerate(printed[TABLE_B5][1:]):
        ds, model = BENCHMARK_DATASETS[i // 3], MODELS[i % 3]
        g = benchmark[(benchmark.dataset == ds) & (benchmark.model == model)]
        columns = [('Original', 'mean_confidence'), ('Original', 'accuracy'),
                   ('Sigmoid', 'mean_confidence'), ('Sigmoid', 'accuracy')]
        for j, (variant, metric) in enumerate(columns):
            check_value(f'Confidence {ds} {model} {variant} {metric}', row[j + 2],
                        100 * g[g.variant == variant][metric].mean())

    # Tables B6a-B6b: match every occupied seed-zero bin using its printed values.
    matched = set()
    for aid, ds in TABLE_B6:
        for row in printed[aid][1:]:
            variant = row[1]
            numbers = re.findall(r'\d+(?:\.\d+)?', row[2])
            bin_number = round(float(numbers[0]) * 10)
            g = bins[(bins.dataset == ds) & (bins.seed == 0) & (bins.variant == variant) & (bins.bin == bin_number)]
            candidates = g[(g['count'] == int(row[3]))
                           & ((g.confidence - float(row[4])).abs() <= .00005001)
                           & ((g.accuracy - float(row[5])).abs() <= .00005001)]
            if len(candidates) != 1:
                mismatches.append(('Reliability bin', ds, row, candidates.index.tolist()))
            else:
                idx = int(candidates.index[0])
                matched.add(idx)
                for metric, col in [('count', 3), ('confidence', 4), ('accuracy', 5)]:
                    check_value(f'Bin {ds} row {idx} {metric}', row[col], candidates.iloc[0][metric])
    assert len(benchmark) == 120
    assert len(bins) == 674

    # Two last rows lack a closing grid rule at the original page breaks, so
    # verify their visible text directly instead of relying on grid extraction.
    source_assets = (VERIFICATION / 'printed_asset_text.txt').read_text(encoding='utf8')
    for idx, row in bins[(bins.seed == 0) & (~bins.index.isin(matched))].iterrows():
        values = [str(int(row['count'])), f'{row.confidence:.4f}', f'{row.accuracy:.4f}']
        assert ' '.join(values) in source_assets
        for metric, printed_value in zip(['count', 'confidence', 'accuracy'], values):
            check_value(f'Page-break bin {row.dataset} {metric}', printed_value, row[metric])
        matched.add(int(idx))
    assert len(matched) == 67
    return matched


def verify_followup(printed, results, paired):
    assert len(results) == 960 and len(paired) == 720

    # Tables C1, C3, C5: complete metric summaries.
    for aid, ds in TABLE_C_MAIN:
        for i, row in enumerate(printed[aid][1:]):
            fraction, method = FRACTIONS[i // 4], METHODS[i % 4]
            g = results[(results.dataset == ds) & (results.fraction == fraction) & (results.method == method)]
            assert len(g) == 20
            for j, (metric, factor) in enumerate([('accuracy', 100), ('log_loss', 1), ('brier', 1), ('ece10', 1)]):
                check_value(f'C main {ds} {fraction} {method} {metric}', row[j + 2], mean_sd(g[metric], factor))

    # Tables C2, C4, C6: confidence and label changes.
    for aid, ds in TABLE_C_CONFIDENCE:
        for i, row in enumerate(printed[aid][1:]):
            fraction, method = FRACTIONS[i // 4], METHODS[i % 4]
            g = results[(results.dataset == ds) & (results.fraction == fraction) & (results.method == method)]
            for j, metric in enumerate(['mean_confidence', 'label_change_rate']):
                check_value(f'C confidence {ds} {fraction} {method} {metric}', row[j + 2], mean_sd(g[metric], 100))

    # Tables C7-C9: paired log-loss changes, range and harms.
    for aid, ds in TABLE_C_PAIRED:
        for i, row in enumerate(printed[aid][1:]):
            fraction, method = FRACTIONS[i // 3], CORRECTIONS[i % 3]
            g = paired[(paired.dataset == ds) & (paired.fraction == fraction) & (paired.method == method)].delta_log_loss
            check_value(f'Paired LL {ds} {fraction} {method}', row[2], mean_sd(g))
            check_value(f'Min {ds} {fraction} {method}', row[3], g.min())
            check_value(f'Max {ds} {fraction} {method}', row[4], g.max())
            check_value(f'Harms {ds} {fraction} {method}', row[5], [int((g > 0).sum()), 20])

    # Table C10: ECE bin-count sensitivity at the full pool.
    for i, row in enumerate(printed[TABLE_C10][1:]):
        ds, method = FOLLOWUP_DATASETS[i // 3], CORRECTIONS[i % 3]
        g = paired[(paired.dataset == ds) & (paired.fraction == 1) & (paired.method == method)]
        for j, metric in enumerate(['delta_ece5', 'delta_ece10', 'delta_ece20']):
            check_value(f'ECE bins {ds} {method} {metric}', row[j + 2], mean_sd(g[metric]))

    # Table 2: digits log loss across calibration sizes.
    for i, row in enumerate(printed[TABLE_2][1:]):
        for j, method in enumerate(METHODS):
            g = results[(results.dataset == 'Digits') & (results.fraction == FRACTIONS[i]) & (results.method == method)].log_loss
            check_value(f'Table 2 {FRACTIONS[i]} {method}', row[j + 1], mean_sd(g))

    # Table 3: paired full-pool changes.
    for i, row in enumerate(printed[TABLE_3][1:]):
        ds, method = FOLLOWUP_DATASETS[i // 3], CORRECTIONS[i % 3]
        g = paired[(paired.dataset == ds) & (paired.fraction == 1) & (paired.method == method)]
        check_value(f'Table3 {ds} {method} LL', row[2], mean_sd(g.delta_log_loss))
        check_value(f'Table3 {ds} {method} BS', row[3], g.delta_brier.mean())
        check_value(f'Table3 {ds} {method} ECE', row[4], g.delta_ece10.mean())
        check_value(f'Table3 {ds} {method} harms', row[5], [int((g.delta_log_loss > 0).sum()), 20])


def verify_saved_predictions(results):
    """Independently recompute every metric from the saved probability arrays."""
    with gzip.open(ROOT / 'predictions.json.gz', 'rt') as stream:
        predictions = json.load(stream)
    assert len(predictions) == 960
    max_difference = 0
    index = results.set_index(['dataset', 'seed', 'fraction', 'method'])
    for record in predictions:
        p = np.asarray(record['probabilities'])
        y = np.asarray(record['labels'])
        assert np.isfinite(p).all()
        assert np.all(p >= 0)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
        row = index.loc[(record['dataset'], record['seed'], record['fraction'], record['method'])]
        for metric, value in score(y, p).items():
            error = abs(value - row[metric])
            max_difference = max(max_difference, float(error))
            assert error < 1e-12

    # Temperature scaling never changes the winning label.
    for ds in FOLLOWUP_DATASETS:
        for seed in range(100, 120):
            for fraction in FRACTIONS:
                group = [r for r in predictions if r['dataset'] == ds and r['seed'] == seed and r['fraction'] == fraction]
                p0 = np.asarray(next(r for r in group if r['method'] == 'Original')['probabilities'])
                pt = np.asarray(next(r for r in group if r['method'] == 'Temperature')['probabilities'])
                assert np.array_equal(p0.argmax(axis=1), pt.argmax(axis=1))
    return predictions, max_difference


def verify_splits():
    """Fitting, calibration and test indices are disjoint and cover each dataset."""
    splits = json.loads((ROOT / 'splits.json').read_text())
    assert len(splits) == 60
    sizes = {'Digits': 1797, 'Iris': 150, 'Wine': 178}
    for r in splits:
        train, test, cal = set(r['train']), set(r['test']), set(r['calibration_order'])
        assert not (train & test or train & cal or test & cal)
        assert len(train) + len(test) + len(cal) == sizes[r['dataset']]
    return splits


def main():
    benchmark = pd.read_csv(ROOT / 'original_benchmark/results.csv')
    bins = pd.read_csv(ROOT / 'original_benchmark/reliability_bins.csv')
    printed_tables = json.load(open(VERIFICATION / 'printed_tables.json', encoding='utf8'))
    printed = {r['aid']: r['tables'][0] for r in printed_tables if r['tables']}

    matched = verify_benchmark(printed, benchmark, bins)
    report = {'benchmark_rows': len(benchmark), 'occupied_bin_rows': len(bins), 'seed_zero_bins_verified': len(matched),
              'numeric_checks': len(checks), 'mismatches': mismatches}
    (VERIFICATION / 'benchmark_verification.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))

    results = pd.read_csv(ROOT / 'followup_results.csv')
    paired = pd.read_csv(ROOT / 'paired_changes.csv')
    verify_followup(printed, results, paired)
    predictions, max_difference = verify_saved_predictions(results)
    splits = verify_splits()

    # Table D3: dataset sizes.
    loaders = [('Breast cancer', load_breast_cancer), ('Digits', load_digits), ('Iris', load_iris), ('Wine', load_wine)]
    for i, (ds, loader) in enumerate(loaders):
        bunch, row = loader(), printed[TABLE_D3][i + 1]
        for c, v in zip([1, 2, 3], [bunch.data.shape[0], bunch.data.shape[1], len(bunch.target_names)]):
            check_value(f'Dataset {ds}', row[c], v)

    report = {'benchmark_rows': len(benchmark), 'followup_rows': len(results), 'paired_change_rows': len(paired),
              'split_records': len(splits), 'probability_records': len(predictions),
              'seed_zero_bins_verified': len(matched), 'numeric_table_checks': len(checks),
              'mismatches': mismatches, 'maximum_saved_metric_recalculation_difference': max_difference}
    (VERIFICATION / 'verification_report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))
    assert not mismatches, 'Printed values differ from the rerun'


if __name__ == '__main__':
    main()
