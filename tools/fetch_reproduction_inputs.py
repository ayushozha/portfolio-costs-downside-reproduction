"""Administrative retrieval and lossless parsing; never train or evaluate a policy."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def fetch():
    expected = json.loads((ROOT / 'evidence/data_provenance/manifest.json').read_text(encoding='utf-8'))
    records = []
    for dataset in expected['datasets']:
        matrix_path = ROOT / dataset['source']['path']
        csv_path = ROOT / dataset['csv_path']
        if matrix_path.exists() or csv_path.exists():
            raise FileExistsError('Use a fresh extraction; input files already exist: ' + dataset['id'])
        with urlopen(Request(dataset['source']['url'], headers={'User-Agent': 'PortfolioReproduction/1.0'}), timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != dataset['source']['sha256']:
            raise ValueError('Pinned matrix hash mismatch: ' + dataset['id'])
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_path.write_bytes(raw)
        matrix = loadmat(str(matrix_path))[dataset['selected_variable']]
        if list(matrix.shape) != [dataset['rows'], dataset['assets']]:
            raise ValueError('Matrix shape mismatch: ' + dataset['id'])
        with csv_path.open('w', newline='', encoding='ascii') as stream:
            writer = csv.writer(stream, lineterminator='\r\n')
            writer.writerow(['asset_{:02d}'.format(index + 1) for index in range(matrix.shape[1])])
            writer.writerows([[format(float(value), '.17g') for value in row] for row in matrix])
        if not np.array_equal(matrix, np.loadtxt(str(csv_path), delimiter=',', skiprows=1)):
            raise ValueError('Lossless conversion failed: ' + dataset['id'])
        if hashlib.sha256(csv_path.read_bytes()).hexdigest() != dataset['csv_sha256']:
            raise ValueError('Converted CSV hash mismatch: ' + dataset['id'])
        records.append({'dataset': dataset['id'], 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'source_url': dataset['source']['url'], 'csv_sha256': dataset['csv_sha256']})
    (ROOT / 'evidence/data_provenance/reproduction_retrieval.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print('Retrieved and hash-verified {} unchanged input matrices.'.format(len(records)))


if __name__ == '__main__':
    fetch()
