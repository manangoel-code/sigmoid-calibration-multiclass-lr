"""Write the dataset snapshots in datasets/ (paper Appendix D4, Table D3).

Each CSV holds a row_index column, the features in loader order and the label.
dataset_metadata.json records sizes, class labels, target names, class counts and
the SHA-256 of each CSV. Files use CRLF line endings, as in the published snapshots,
so the checksums are identical on every platform.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

FOLDER = Path(__file__).resolve().parent / 'datasets'
DATASETS = [('breast_cancer', load_breast_cancer), ('digits', load_digits), ('iris', load_iris), ('wine', load_wine)]


def main():
    FOLDER.mkdir(exist_ok=True)
    metadata = {}
    for key, loader in DATASETS:
        bunch = loader()
        frame = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        frame.insert(0, 'row_index', np.arange(len(frame)))
        frame['label'] = bunch.target
        content = frame.to_csv(index=False, lineterminator='\r\n').encode()
        (FOLDER / f'{key}.csv').write_bytes(content)
        labels, counts = np.unique(bunch.target, return_counts=True)
        metadata[key] = {
            'rows': int(bunch.data.shape[0]),
            'features': int(bunch.data.shape[1]),
            'feature_names': list(bunch.feature_names),
            'class_labels': labels.tolist(),
            'target_names': [str(name) for name in bunch.target_names],
            'class_counts': {str(label): int(count) for label, count in zip(labels, counts)},
            'sha256': hashlib.sha256(content).hexdigest(),
        }
        print(key, metadata[key]['rows'], 'rows', metadata[key]['sha256'])
    (FOLDER / 'dataset_metadata.json').write_bytes(json.dumps(metadata, indent=2).replace('\n', '\r\n').encode())


if __name__ == '__main__':
    main()
