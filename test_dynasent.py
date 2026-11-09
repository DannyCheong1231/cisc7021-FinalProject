import json
import pytest
import torch
from sst5 import Head, scores


def test_three_class_head_and_metrics():
    names = ['負面', '中性', '正面']
    assert Head(8, 4, num_classes=3)(torch.randn(2, 8)).shape == (2, 3)
    result = scores(torch.arange(3), torch.arange(3), names)
    assert result['macro_f1'] == 1
    assert len(result['confusion_matrix']) == 3
    assert [r['label'] for r in result['per_class']] == names


def test_dynasent_gold_filter_and_round_boundaries(tmp_path):
    from sst5 import read_dynasent
    for round_id, kind in [('01', 'yelp'), ('02', 'dynabench')]:
        for part in ['train', 'dev', 'test']:
            records = [dict(sentence=f'{round_id} {part} {i}', gold_label=label,
                            model_0_label='negative')
                       for i, label in enumerate(['negative', 'neutral', 'positive', 'mixed', None])]
            path = tmp_path / f'dynasent-v1.1-round{round_id}-{kind}-{part}.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in records), 'utf-8')
    splits, stats = read_dynasent(tmp_path)
    assert {k:len(v) for k,v in splits.items()} == {'train':6, 'validation':6, 'test':6}
    assert [y for _,y in splits['train']] == [0,1,2,0,1,2]
    assert stats['test_round_sizes'] == [3,3]
    assert stats['excluded_counts']['round01-train'] == 2


def test_three_class_training_checkpoint_and_round_metrics(tmp_path):
    from sst5 import fit_head, DYNASENT_LABELS
    config = dict(input_size=3, hidden_size=4, lr=.0001, batch_size=3,
                  epochs=2, labels=DYNASENT_LABELS, dataset='dynasent',
                  dataset_stats={'test_round_sizes':[3,3]})
    data = {name:(torch.eye(3), torch.arange(3)) for name in ['train','validation']}
    data['test'] = (torch.eye(3).repeat(2,1), torch.arange(3).repeat(2))
    result, head = fit_head(data, config, tmp_path)
    restored = Head(3, 4, num_classes=3).eval()
    restored.load_state_dict(torch.load(tmp_path/'head.pt',weights_only=True)['state_dict'])
    torch.testing.assert_close(head(data['test'][0]), restored(data['test'][0]))
    assert len(result['test']['confusion_matrix']) == 3
    for value in result['test_by_round'].values():
        assert sum(c['support'] for c in value['per_class']) == 3
