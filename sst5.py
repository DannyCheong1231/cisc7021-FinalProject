"""Frozen Llama features -> trainable SST-5 head. No backbone fine-tuning."""
import argparse
import csv
import json
import random
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

LABELS = ['非常負面', '負面', '中性', '正面', '非常正面']
DATA_REVISION = 'fa0a89e0cabf5eac3ff4e1d27551fad4046ec012'
COURSE_MODEL = 'rzzhan/tiny-llama-stories-42m'
COURSE_REVISION = 'eefdbd1577163cb01d906066bbe9176da71f86a1'


def parse_tree(line):
    root = re.match(r'^\(([0-4])\s', line)
    leaves = re.findall(r'\([0-4] ([^()\s]+)\)', line)
    if root is None or not leaves:
        raise ValueError('Expected a five-class SST tree with a sentence-root label')
    escapes = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']',
               '-LCB-': '{', '-RCB-': '}'}
    text = ' '.join(escapes.get(word, word).replace('\\/', '/') for word in leaves)
    return text, int(root[1])


def load_sst(data_dir):
    data_dir = Path(data_dir) / DATA_REVISION
    data_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split, filename in [('train', 'train.txt'), ('validation', 'dev.txt'),
                            ('test', 'test.txt')]:
        path = data_dir / filename
        if not path.exists():
            url = (f'https://raw.githubusercontent.com/stanfordnlp/'
                   f'sentiment-treebank/{DATA_REVISION}/fiveclass/{filename}')
            content = urllib.request.urlopen(url, timeout=60).read()
            # Parse before writing, so failed downloads never become a cached dataset.
            for line in content.decode('utf-8').splitlines():
                if line.strip():
                    parse_tree(line)
            path.write_bytes(content)
        splits[split] = [parse_tree(line) for line in path.read_text('utf-8').splitlines()
                         if line.strip()]
    return splits


def last_token(hidden, attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=hidden.device)
    indices = positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(1).values
    if (indices < 0).any():
        raise ValueError('Every input must contain at least one non-padding token')
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


class FrozenEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        # Match positions across left/right padding, including single-example inference.
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=position_ids, use_cache=False)
        return last_token(output.last_hidden_state, attention_mask).float()


class Head(nn.Sequential):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        layers = [nn.LayerNorm(input_size)]
        if hidden_size:
            layers += [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        super().__init__(*layers, nn.Linear(hidden_size or input_size, 5))


def load_encoder(model_id, revision, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    # float32 is small enough for 42M; larger GPU backbones use float16.
    dtype = torch.float32 if model_id == COURSE_MODEL or device == 'cpu' else torch.float16
    backbone = AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
    return tokenizer, FrozenEncoder(backbone.to(device))


def extract(encoder, tokenizer, texts, batch_size, max_length, device):
    features = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start:start + batch_size], padding=True,
                           truncation=True, max_length=max_length, return_tensors='pt')
        tokens = {k: v.to(device) for k, v in tokens.items() if k in ('input_ids', 'attention_mask')}
        features.append(encoder(**tokens).cpu())
        if start % (batch_size * 50) == 0:
            print(f'  features {min(start + batch_size, len(texts))}/{len(texts)}', flush=True)
    return torch.cat(features)


def scores(labels, predictions):
    matrix = torch.bincount(labels.cpu() * 5 + predictions.cpu(), minlength=25).reshape(5, 5)
    tp = matrix.diag().float()
    precision = tp / matrix.sum(0).clamp(min=1)
    recall = tp / matrix.sum(1).clamp(min=1)
    f1 = 2 * tp / (matrix.sum(0) + matrix.sum(1)).clamp(min=1)
    return {'accuracy': (tp.sum() / matrix.sum()).item(), 'macro_f1': f1.mean().item(),
            'confusion_matrix': matrix.tolist(),
            'per_class': [{'label': LABELS[i], 'precision': precision[i].item(),
                           'recall': recall[i].item(), 'f1': f1[i].item(),
                           'support': matrix[i].sum().item()} for i in range(5)]}


@torch.no_grad()
def evaluate(head, features, labels):
    head.eval()
    return scores(labels, head(features).argmax(-1))


def train(args, epoch_callback=None):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output or f'runs/train-{datetime.now():%Y%m%d-%H%M%S-%f}')
    out.mkdir(parents=True, exist_ok=True)
    # Never replace the provenance/results of an earlier experiment silently.
    if (out / 'head.pt').exists():
        raise ValueError('Output already contains a trained head; choose a new --output directory')
    splits = load_sst(args.data_dir)
    if getattr(args, 'features', None):
        source = Path(args.features).resolve()
        cached = torch.load(source, map_location='cpu', weights_only=True)
        config = dict(cached['config'])
        if config['data_revision'] != DATA_REVISION or config['labels'] != LABELS:
            raise ValueError('Cached features must use the same SST revision and label order')
        data = cached['splits']
        for name, rows in splits.items():
            features, labels = data[name]
            expected = torch.tensor([label for _, label in rows])
            if features.shape != (len(rows), config['input_size']) or not torch.equal(labels, expected):
                raise ValueError(f'Cached {name} features/labels do not match the SST split')
        config.update(command='train', output=str(out.resolve()), lr=args.lr, seed=args.seed,
                      epochs=args.epochs, hidden_size=args.hidden_size, batch_size=args.batch_size,
                      dropout=getattr(args, 'dropout', .3), weight_decay=getattr(args, 'weight_decay', .1),
                      data_dir=args.data_dir, device='cpu', feature_source=str(source),
                      source_feature_seconds=config.get('feature_seconds'), feature_seconds=0,
                      peak_gpu_memory_bytes=None,
                      seed_scope='reset immediately before head initialization')
        print(f'Reusing frozen features: {source}\nLearning rate: {args.lr}; seed: {args.seed}', flush=True)
    else:
        revision = args.revision or (COURSE_REVISION if args.model == COURSE_MODEL else 'main')
        tokenizer, encoder = load_encoder(args.model, revision, args.device)
        if not 2 <= args.max_length <= encoder.backbone.config.max_position_embeddings:
            raise ValueError('max-length must fit the backbone context and allow a content token')
        resolved_revision = getattr(encoder.backbone.config, '_commit_hash', None) or revision
        config = dict(vars(args), output=str(out.resolve()), revision=resolved_revision,
                      data_revision=DATA_REVISION, labels=LABELS,
                      pooling='last_non_padding_token', add_eos_token=False)
        config['split_sizes'] = {name: len(rows) for name, rows in splits.items()}
        config['class_counts'] = {name: torch.bincount(torch.tensor([y for _, y in rows]),
                                 minlength=5).tolist() for name, rows in splits.items()}
        config['frozen_parameters'] = sum(p.numel() for p in encoder.parameters())
        config['input_size'] = encoder.backbone.config.hidden_size
        start = time.perf_counter()
        data = {}
        for name, rows in splits.items():
            print(f'Extracting {name}: {len(rows)} sentences', flush=True)
            texts, labels = zip(*rows)
            data[name] = (extract(encoder, tokenizer, list(texts), args.feature_batch_size,
                                  args.max_length, args.device), torch.tensor(labels))
        config['feature_seconds'] = time.perf_counter() - start
        config['peak_gpu_memory_bytes'] = (torch.cuda.max_memory_allocated()
                                           if args.device == 'cuda' else None)
        del encoder
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    torch.save({'config': config, 'splits': data}, out / 'features.pt')
    results, head = fit_head(data, config, out, epoch_callback)
    with torch.no_grad():
        predicted = head(data['test'][0]).argmax(-1).tolist()
    with (out / 'test_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'gold_id', 'predicted_id', 'gold_level', 'predicted_level'])
        for (text, gold), pred in zip(splits['test'], predicted):
            writer.writerow([text, gold, pred, gold + 1, pred + 1])
    print(json.dumps({'best_epoch': results['best_epoch'], 'test': results['test']}, ensure_ascii=False), flush=True)
    print(f'Results: {out.resolve()}', flush=True)
    print(f'.venv\\Scripts\\python.exe verify_results.py --run "{out.resolve()}"', flush=True)


def fit_head(data, config, out, epoch_callback=None, evaluate_test=True):
    """Train from frozen features; tuning can omit the test split entirely."""
    config.setdefault('dropout', .3)
    config.setdefault('weight_decay', .1)
    args = argparse.Namespace(**config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'head.pt').exists():
        raise ValueError('Output already contains a trained head; choose a new output directory')
    # Cached features are small: train the head on CPU, independently of the LLM.
    head = Head(config['input_size'], args.hidden_size, args.dropout)
    config['trainable_parameters'] = sum(p.numel() for p in head.parameters())
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(*data['train']), batch_size=args.batch_size, shuffle=True)
    history, best = [], -1
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(features), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(labels)
        validation = evaluate(head, *data['validation'])
        training = evaluate(head, *data['train'])
        with torch.no_grad():
            validation_loss = nn.functional.cross_entropy(
                head(data['validation'][0]), data['validation'][1]).item()
        record = dict(epoch=epoch, train_loss=total / len(data['train'][1]),
                      validation_loss=validation_loss,
                      train_accuracy=training['accuracy'], train_macro_f1=training['macro_f1'],
                      validation_accuracy=validation['accuracy'],
                      validation_macro_f1=validation['macro_f1'])
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation['macro_f1'] > best:
            best = validation['macro_f1']
            torch.save({'state_dict': head.state_dict(), 'config': config, 'epoch': epoch}, out / 'head.pt')
        if epoch_callback is not None:
            epoch_callback(history)
    checkpoint = torch.load(out / 'head.pt', weights_only=True)
    head.load_state_dict(checkpoint['state_dict'])
    results = {'config': config, 'best_epoch': checkpoint['epoch'], 'history': history,
               'head_training_seconds': time.perf_counter() - start,
               'validation': evaluate(head, *data['validation'])}
    if evaluate_test:
        results['test'] = evaluate(head, *data['test'])
    (out / 'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), 'utf-8')
    return results, head


def predict(args):
    if not args.text.strip():
        raise ValueError('Review text must not be empty')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    tokenizer, encoder = load_encoder(cfg['model'], cfg['revision'], args.device)
    head = Head(cfg['input_size'], cfg['hidden_size'], cfg.get('dropout', .3)).eval()
    head.load_state_dict(checkpoint['state_dict'])
    features = extract(encoder, tokenizer, [args.text], 1, cfg['max_length'], args.device)
    with torch.no_grad():
        probabilities = head(features).softmax(-1)[0]
    label = probabilities.argmax().item()
    print(json.dumps({'label_id': label, 'level': label + 1, 'sentiment': cfg['labels'][label],
                      'probabilities': probabilities.tolist()}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--model', default=COURSE_MODEL)
    train_parser.add_argument('--revision')
    train_parser.add_argument('--output', help='Default: new timestamped directory under runs/')
    train_parser.add_argument('--features', help='Reuse features.pt; model/tokenizer settings come from the cache')
    train_parser.add_argument('--data-dir', default='data')
    train_parser.add_argument('--epochs', type=int, default=30)
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--hidden-size', type=int, default=128, help='0: LayerNorm + linear head')
    train_parser.add_argument('--dropout', type=float, default=.3)
    train_parser.add_argument('--weight-decay', type=float, default=.1)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--feature-batch-size', type=int, default=32)
    train_parser.add_argument('--max-length', type=int, default=128)
    train_parser.add_argument('--seed', type=int, default=7021)
    prediction = sub.add_parser('predict')
    prediction.add_argument('--checkpoint', default='runs/sst5-42m/head.pt')
    prediction.add_argument('--text', required=True)
    for command in [train_parser, prediction]:
        command.add_argument('--device', choices=['cpu', 'cuda'],
                             default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.command == 'train':
        if min(args.epochs, args.batch_size, args.feature_batch_size) < 1 or args.hidden_size < 0 or args.lr <= 0:
            parser.error('Epochs, batch sizes and lr must be positive; hidden-size must be nonnegative')
        if not 0 <= args.dropout < 1 or args.weight_decay < 0:
            parser.error('dropout must be in [0,1); weight-decay must be nonnegative')
        train(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
