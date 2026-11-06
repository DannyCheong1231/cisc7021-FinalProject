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
    head = Head(cfg['input_size'], cfg['hidden_size'], cfg.get('dropout', .3)).eval()
    head.load_state_dict(checkpoint['state_dict'])
    features, gold = cached['splits']['test']
    with torch.no_grad():
        predictions = head(features).argmax(-1).tolist()
    with (run / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(gold) == cfg['split_sizes']['test']
    matrix = [[0] * 5 for _ in range(5)]
    for expected, predicted, row in zip(gold.tolist(), predictions, rows):
        assert expected == int(row['gold_id'])
        assert predicted == int(row['predicted_id'])
        assert int(row['gold_level']) == expected + 1
        assert int(row['predicted_level']) == predicted + 1
        matrix[expected][predicted] += 1
    correct = sum(matrix[i][i] for i in range(5))
    accuracy = correct / len(rows)
    f1s = []
    for i in range(5):
        denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
        f1s.append(2 * matrix[i][i] / denominator if denominator else 0.0)
    macro_f1 = sum(f1s) / 5
    assert matrix == metrics['test']['confusion_matrix']
    assert abs(accuracy - metrics['test']['accuracy']) < 1e-6
    assert abs(macro_f1 - metrics['test']['macro_f1']) < 1e-6
    best = max(metrics['history'], key=lambda row: row['validation_macro_f1'])
    assert checkpoint['epoch'] == best['epoch'] == metrics['best_epoch']
    assert sum(p.numel() for p in head.parameters()) == cfg['trainable_parameters']
    print(f'PASS: loaded saved head and recomputed all {len(rows)} test predictions.')
    print('PASS: CSV predictions, 0-4 to 1-5 mapping, confusion matrix and metrics agree.')
    print(f'PASS: checkpoint selected by validation macro-F1: epoch {best["epoch"]}.')
    print(f'Correct: {correct}/{len(rows)}; accuracy: {accuracy:.6f}; macro-F1: {macro_f1:.6f}')
    print('Scope: uses cached LLM features; run pytest to check encoder freeze/padding,')
    print('or run training with a new output directory to repeat the whole pipeline.')


if __name__ == '__main__':
    main()
