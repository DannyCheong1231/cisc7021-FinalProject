import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from sst5 import LABELS, FrozenEncoder, Head, last_token, parse_tree, scores


def test_sentence_root_labels_not_leaf_labels():
    for label in range(5):
        text, actual = parse_tree(f'({label} (4 Great) (0 (2 -LRB-) (1 film)))')
        assert text == 'Great ( film'
        assert actual == label
    assert LABELS == ['非常負面', '負面', '中性', '正面', '非常正面']
    with pytest.raises(ValueError):
        parse_tree('(5 (2 movie))')


def test_last_token_ignores_left_and_right_padding():
    hidden = torch.arange(24).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]])
    assert torch.equal(last_token(hidden, mask), hidden[[0, 1], [1, 3]])
    with pytest.raises(ValueError):
        last_token(hidden, torch.zeros_like(mask))


def test_backbone_frozen_and_head_learns():
    torch.manual_seed(7021)
    backbone = LlamaModel(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2))
    encoder = FrozenEncoder(backbone)
    encoder.train()
    assert not backbone.training
    before = {k: v.clone() for k, v in backbone.state_dict().items()}
    ids = torch.tensor([[1, 3, 4], [1, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    features = encoder(input_ids=ids, attention_mask=mask)
    assert features.shape == (2, 16)
    assert not features.requires_grad
    head = Head(16, 16)
    original = head(features).detach().clone()
    opt = torch.optim.AdamW(head.parameters(), lr=0.01)
    for _ in range(40):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(features), torch.tensor([0, 4]))
        loss.backward()
        opt.step()
    head.eval()
    assert head(features).argmax(-1).tolist() == [0, 4]
    assert not torch.equal(original, head(features))
    assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
    assert all(torch.equal(before[k], v) for k, v in backbone.state_dict().items())
    # A sentence should have the same features alone and in a padded batch.
    single = encoder(input_ids=ids[1:2, :2], attention_mask=mask[1:2, :2])
    torch.testing.assert_close(features[1:2], single, atol=1e-5, rtol=1e-5)
    left = encoder(input_ids=torch.tensor([[0, 1, 5]]),
                   attention_mask=torch.tensor([[0, 1, 1]]))
    torch.testing.assert_close(single, left, atol=1e-5, rtol=1e-5)


def test_metrics_include_all_five_classes_and_confusion_orientation():
    result = scores(torch.tensor([0, 4]), torch.tensor([0, 0]))
    assert result['accuracy'] == 0.5
    assert result['macro_f1'] == pytest.approx((2 / 3) / 5)
    assert result['confusion_matrix'][4][0] == 1


def test_head_checkpoint_round_trip(tmp_path):
    for hidden_size in (0, 16):
        head = Head(16, hidden_size).eval()
        features = torch.randn(3, 16)
        path = tmp_path / 'head.pt'
        torch.save({'state_dict': head.state_dict()}, path)
        restored = Head(16, hidden_size).eval()
        restored.load_state_dict(torch.load(path, weights_only=True)['state_dict'])
        torch.testing.assert_close(head(features), restored(features))


def test_training_records_curves_and_calls_notebook_callback(tmp_path, monkeypatch):
    from argparse import Namespace
    import sst5
    backbone = LlamaModel(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2))
    monkeypatch.setattr(sst5, 'load_encoder', lambda *a: (None, FrozenEncoder(backbone)))
    monkeypatch.setattr(sst5, 'load_sst', lambda *a: {
        split: [('movie', i) for i in range(5)] for split in ['train', 'validation', 'test']})
    monkeypatch.setattr(sst5, 'extract', lambda *a: torch.eye(5, 16))
    seen = []
    args = Namespace(output=str(tmp_path), revision=None, model=sst5.COURSE_MODEL,
        data_dir='data', device='cpu', max_length=16, feature_batch_size=5,
        hidden_size=8, batch_size=5, lr=0.001, epochs=2, seed=7021)
    sst5.train(args, epoch_callback=lambda history: seen.append(dict(history[-1])))
    assert len(seen) == 2
    for row in seen:
        assert row['validation_loss'] > 0
        assert 0 <= row['train_accuracy'] <= 1
        assert 0 <= row['train_macro_f1'] <= 1


def test_cached_head_training_does_not_evaluate_test_when_tuning(tmp_path):
    from sst5 import fit_head
    cfg = dict(seed=7021, input_size=5, hidden_size=8, batch_size=5,
               lr=0.001, epochs=2)
    # No test split: tuning must need only train and validation.
    data = {name: (torch.eye(5), torch.arange(5)) for name in ['train', 'validation']}
    torch.manual_seed(7021)
    first, _ = fit_head(data, dict(cfg), tmp_path / 'first', evaluate_test=False)
    torch.manual_seed(7021)
    second, _ = fit_head(data, dict(cfg), tmp_path / 'second', evaluate_test=False)
    assert first['history'] == second['history']
    assert 'test' not in first
    assert first['best_epoch'] == max(first['history'], key=lambda r:r['validation_macro_f1'])['epoch']


def test_train_from_features_skips_backbone_and_preserves_provenance(tmp_path, monkeypatch):
    import json
    from argparse import Namespace
    import sst5
    source = tmp_path / 'features.pt'
    rows = [('movie', i) for i in range(5)]
    config = dict(model='cached-llama', revision='fixed-revision', data_revision=sst5.DATA_REVISION,
                  labels=LABELS, input_size=5, max_length=128,
                  split_sizes={name: 5 for name in ['train', 'validation', 'test']})
    data = {name: (torch.eye(5), torch.arange(5)) for name in ['train', 'validation', 'test']}
    torch.save({'config': config, 'splits': data}, source)
    monkeypatch.setattr(sst5, 'load_sst', lambda *a: {name: rows for name in data})
    def forbidden(*a):
        pytest.fail('Cached-feature training must not load the backbone')
    monkeypatch.setattr(sst5, 'load_encoder', forbidden)
    output = tmp_path / 'trained'
    args = Namespace(features=str(source), output=str(output), seed=7021, epochs=2,
                     lr=0.0001, hidden_size=8, batch_size=5, data_dir='data')
    sst5.train(args)
    metrics = json.loads((output / 'metrics.json').read_text('utf-8'))
    assert metrics['config']['model'] == 'cached-llama'
    assert metrics['config']['lr'] == 0.0001
    assert metrics['config']['revision'] == 'fixed-revision'
    assert metrics['config']['feature_seconds'] == 0
    assert (output / 'features.pt').exists()
    assert (output / 'test_predictions.csv').exists()
    # Reusing an output must fail without overwriting a completed experiment.
    with pytest.raises(ValueError, match='already contains'):
        sst5.train(args)


def test_fit_head_uses_and_records_regularization(tmp_path, monkeypatch):
    from sst5 import fit_head
    actual_adamw = torch.optim.AdamW
    used = []
    def optimizer(*args, **kwargs):
        used.append(kwargs['weight_decay'])
        return actual_adamw(*args, **kwargs)
    monkeypatch.setattr(torch.optim, 'AdamW', optimizer)
    config = dict(input_size=5, hidden_size=8, batch_size=5, lr=.0001,
                  epochs=2, dropout=.6, weight_decay=.2)
    data = {name: (torch.eye(5), torch.arange(5)) for name in ['train', 'validation']}
    result, head = fit_head(data, config, tmp_path, evaluate_test=False)
    assert used == [.2]
    assert next(m for m in head.modules() if isinstance(m, torch.nn.Dropout)).p == .6
    saved = torch.load(tmp_path / 'head.pt', weights_only=True)['config']
    assert saved['dropout'] == result['config']['dropout'] == .6
    assert saved['weight_decay'] == result['config']['weight_decay'] == .2
