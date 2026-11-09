"""Independently verify saved SST-5 results without retraining or downloading."""
import argparse
import csv
import json
from pathlib import Path

import torch

from sst5 import Head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/sst5-42m')
    args = parser.parse_args()
    run = Path(args.run)
    metrics = json.loads((run / 'metrics.json').read_text('utf-8'))
    checkpoint = torch.load(run / 'head.pt', map_location='cpu', weights_only=True)
    cached = torch.load(run / 'features.pt', map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    n = len(cfg['labels'])
    head = Head(cfg['input_size'], cfg['hidden_size'], cfg.get('dropout', .3), n).eval()
    head.load_state_dict(checkpoint['state_dict'])
    features, gold = cached['splits']['test']
    with torch.no_grad():
        predictions = head(features).argmax(-1).tolist()
    with (run / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(gold) == cfg['split_sizes']['test']
    matrix = [[0] * n for _ in range(n)]
    for expected, predicted, row in zip(gold.tolist(), predictions, rows):
        assert expected == int(row['gold_id'])
        assert predicted == int(row['predicted_id'])
        assert int(row['gold_level']) == expected + 1
        assert int(row['predicted_level']) == predicted + 1
        matrix[expected][predicted] += 1
    correct = sum(matrix[i][i] for i in range(n))
    accuracy = correct / len(rows)
    f1s = []
    for i in range(n):
        denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
        f1s.append(2 * matrix[i][i] / denominator if denominator else 0.0)
    macro_f1 = sum(f1s) / n
    assert matrix == metrics['test']['confusion_matrix']
    assert abs(accuracy - metrics['test']['accuracy']) < 1e-6
    assert abs(macro_f1 - metrics['test']['macro_f1']) < 1e-6
    if cfg.get('dataset') == 'dynasent':
        offset = 0
        for name, size in zip(['round1', 'round2'], cfg['dataset_stats']['test_round_sizes']):
            local = [[0] * n for _ in range(n)]
            for row in rows[offset:offset+size]:
                local[int(row['gold_id'])][int(row['predicted_id'])] += 1
            result = metrics['test_by_round'][name]
            assert local == result['confusion_matrix']
            acc = sum(local[i][i] for i in range(n)) / size
            local_f1 = []
            for i in range(n):
                denominator = sum(local[i]) + sum(r[i] for r in local)
                local_f1.append(2 * local[i][i] / denominator if denominator else 0.0)
            assert abs(acc - result['accuracy']) < 1e-6
            assert abs(sum(local_f1)/n - result['macro_f1']) < 1e-6
            offset += size
        assert offset == len(rows)
        print('PASS: Round 1 and Round 2 test boundaries, confusion matrices and metrics agree.')
    best = max(metrics['history'], key=lambda row: row['validation_macro_f1'])
    assert checkpoint['epoch'] == best['epoch'] == metrics['best_epoch']
    assert sum(p.numel() for p in head.parameters()) == cfg['trainable_parameters']
    print(f'PASS: loaded saved head and recomputed all {len(rows)} test predictions.')
    print(f'PASS: CSV predictions, 0-{n-1} to 1-{n} mapping, confusion matrix and metrics agree.')
    print(f'PASS: checkpoint selected by validation macro-F1: epoch {best["epoch"]}.')
    print(f'Correct: {correct}/{len(rows)}; accuracy: {accuracy:.6f}; macro-F1: {macro_f1:.6f}')
    print('Scope: uses cached LLM features; run pytest to check encoder freeze/padding,')
    print('or run training with a new output directory to repeat the whole pipeline.')


if __name__ == '__main__':
    main()
