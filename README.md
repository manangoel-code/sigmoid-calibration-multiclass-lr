# When Sigmoid Calibration Worsens Multiclass Logistic Regression

Code, data snapshots and full results for the paper

> **When Sigmoid Calibration Worsens Multiclass Logistic Regression: Effects of Calibration Set Size and Score Representation**
> Manan Goel, 2026. [Read the paper (PDF)](paper/When_Sigmoid_Calibration_Worsens_Multiclass_Logistic_Regression.pdf)

Sigmoid (Platt) calibration is meant to improve predicted probabilities. For multinomial logistic regression it
made them worse in these experiments. A preliminary benchmark on breast cancer and digits, with logistic
regression, random forest and naive Bayes, showed the problem. A controlled follow-up on digits, iris and wine
then varied the calibration set size and compared three corrections: sigmoid calibration on logits, sigmoid
calibration on class-versus-rest log odds, and temperature scaling.

On digits the damage from sigmoid calibration shrank as the calibration set grew but did not disappear. On iris
and wine it was still there with the full calibration pool. Fitting the sigmoid on log odds reduced the damage.
On digits, temperature scaling came close to the uncalibrated model with the full pool, but it was more
sensitive to small calibration sets.

## Repository layout

The file paths match Table D1 of the paper.

```
paper/                                  the paper (PDF)
original_benchmark/
    experiment.py                       preliminary benchmark (Sections 2.1, 3.1; Appendix B)
    make_graph.py                       redraws the reliability plot (optional, needs matplotlib)
    recovered_original_experiment.py    original combined analysis + plot script, kept for provenance
    results.csv                         120 evaluation rows
    summary.csv                         12 grouped means and sample SDs
    reliability_bins.csv                674 occupied reliability-bin records
    reliability.png                     original project copy of the reliability graph
investigate.py                          controlled follow-up (Sections 2.3-2.5, 3.2-3.5; Appendix C)
followup_results.csv                    960 follow-up evaluation rows
followup_summary.csv                    48 grouped means and sample SDs
paired_changes.csv                      720 corrected-minus-original rows
predictions.json.gz                     every test probability vector (960 records)
splits.json                             fitting/test indices and nested calibration order (60 records)
environment.json                        package versions recorded by investigate.py
export_datasets.py                      writes the dataset snapshots below
datasets/                               breast_cancer, digits, iris, wine snapshots + dataset_metadata.json
verify_results.py                       checks the paper's printed numbers against all records
verification/                           table values extracted from the paper and the verification reports
*.png                                   figures used in the paper (Figures 1, 2, B1, B2)
```

## Reproduce the results

You need Python 3.12 and the pinned versions in `requirements.txt`. The scikit-learn version matters because the
follow-up uses its private `_SigmoidCalibration` class.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

python original_benchmark/experiment.py   # ~15 s, rewrites the benchmark CSVs
python investigate.py                     # ~30 s, rewrites the follow-up records
python export_datasets.py                 # rewrites datasets/ (checksums must match)
python verify_results.py                  # checks everything below
```

Each script writes its outputs to its own folder, so you can run it from anywhere. With the pinned versions on
Windows, a rerun reproduces every committed CSV, JSON and gzip file byte for byte. The scripts write fixed CRLF
line endings, so other operating systems should match as well; I have only tested Windows.

`verify_results.py` checks:

- 1,391 numbers printed in the paper's tables, at their printed precision (zero mismatches)
- all 67 seed-zero reliability bins and all 674 occupied-bin records
- all 960 saved probability vectors, re-scored independently (maximum difference 1.1e-16)
- that the 60 fitting/calibration/test allocations are disjoint and complete
- that temperature scaling never changes a predicted label (240 conditions)

It writes its reports to `verification/benchmark_verification.json` and `verification/verification_report.json`.

To redraw the original reliability plot, install `matplotlib>=3.10` and run
`python original_benchmark/make_graph.py`. It writes `reliability_redrawn.png`; the archived `reliability.png`
and the other paper figures are the PNG files used in the paper.

## Data

All four datasets ship with scikit-learn (`load_breast_cancer`, `load_digits`, `load_iris`, `load_wine`), so
nothing needs downloading. The snapshots in `datasets/` record the exact rows used. Their `row_index` column
matches the indices in `splits.json` and `predictions.json.gz`. Original sources:

- Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
- Alpaydin, E., & Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49
- Fisher, R. A. (1936). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76
- Aeberhard, S., & Forina, M. (1992). Wine. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J

## Scope

The paper has not been peer reviewed. `verify_results.py` shows that its numbers follow from this code and data.
Whether the findings hold for other models or datasets is a separate question that these checks do not answer.

## Citation

See [`CITATION.cff`](CITATION.cff), or cite as:

Goel, M. (2026). *When Sigmoid Calibration Worsens Multiclass Logistic Regression: Effects of Calibration Set Size
and Score Representation*.
