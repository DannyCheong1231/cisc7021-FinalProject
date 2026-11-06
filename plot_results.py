"""Render actual saved training results as standalone PNG charts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'


def plot_history(history, title, best_epoch):
    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [('train_loss', 'validation_loss', 'Cross-entropy loss'),
             ('train_accuracy', 'validation_accuracy', 'Accuracy'),
             ('train_macro_f1', 'validation_macro_f1', 'Macro-F1')]
    for ax, (train_key, val_key, label) in zip(axes, pairs):
        for key, name in [(train_key, 'Train'), (val_key, 'Validation')]:
            ax.plot(epochs, [row[key] for row in history], label=name, linewidth=2)
        ax.axvline(best_epoch, color='grey', linestyle='--', label=f'Selected: {best_epoch}')
        ax.set(xlabel='Epoch', ylabel=label)
        if label != 'Cross-entropy loss':
            ax.set_ylim(0, 1)
        ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    return fig


def plot_sweep_losses(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / 'summary.json').read_text('utf-8'))
    trials = summary['trials']
    fig, axes = plt.subplots(1, len(trials), figsize=(5 * len(trials), 4),
                             sharey=True, squeeze=False)
    for ax, trial in zip(axes[0], trials):
        result = json.loads((run_dir / Path(trial['output']).name / 'metrics.json').read_text('utf-8'))
        history = result['history']
        epochs = [row['epoch'] for row in history]
        ax.plot(epochs, [row['train_loss'] for row in history], label='Train loss', linewidth=2)
        ax.plot(epochs, [row['validation_loss'] for row in history], label='Validation loss', linewidth=2)
        ax.axvline(result['best_epoch'], color='grey', linestyle='--',
                   label=f"Selected by val F1: {result['best_epoch']}")
        ax.set(xlabel='Epoch', title=f"Learning rate = {trial['lr']:g}")
        ax.legend(fontsize=9); ax.grid(alpha=.2)
    axes[0, 0].set_ylabel('Cross-entropy loss')
    fig.suptitle(f"Frozen Llama + MLP | Train vs validation loss | seed {summary['seed']}")
    fig.tight_layout()
    path = run_dir / 'train_validation_losses.png'
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('Chart:', path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--sweep', help='Plot train/validation losses from an existing LR sweep')
    inputs.add_argument('--run', help='Plot training curves from one completed experiment')
    args = parser.parse_args()
    if args.sweep:
        plot_sweep_losses(args.sweep)
        return
    if args.run:
        run = Path(args.run)
        result = json.loads((run / 'metrics.json').read_text('utf-8'))
        config = result['config']
        fig = plot_history(result['history'],
                           f"LR={config['lr']:g}, seed={config['seed']}", result['best_epoch'])
        path = run / 'training.png'
        fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
        print('Chart:', path.resolve())
        return
    OUT.mkdir(exist_ok=True)
    font = Path('C:/Windows/Fonts/msjh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'figure.dpi': 150, 'axes.spines.top': False, 'axes.spines.right': False})
    runs = {'42M': ROOT / 'runs/sst5-42m-curves', '1.1B': ROOT / 'runs/sst5-1.1b'}
    results = {name: json.loads((path/'metrics.json').read_text('utf-8'))
               for name, path in runs.items() if (path/'metrics.json').exists()}
    if not results:
        raise FileNotFoundError('No completed training results found')
    for name, result in results.items():
        fig = plot_history(result['history'], f'{name} frozen Llama + MLP | SST-5, seed 7021',
                           result['best_epoch'])
        fig.savefig(OUT / f'training_{name}.png', bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5), squeeze=False)
    labels = ['非常負面', '負面', '中性', '正面', '非常正面']
    for ax, (name, r) in zip(axes[0], results.items()):
        matrix = np.array(r['test']['confusion_matrix'])
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{matrix[i,j]}\n{normalized[i,j]:.0%}', ha='center', va='center',
                        fontsize=10, color='white' if normalized[i,j] > .5 else 'black')
        ax.set_xticks(range(5), labels, rotation=30, ha='right'); ax.set_yticks(range(5), labels)
        ax.set(xlabel='預測情感', ylabel='真實情感', title=f'{name}: Test confusion matrix')
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.tight_layout(); fig.savefig(OUT/'confusion_matrices.png', bbox_inches='tight'); plt.close(fig)
    names = list(results); x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, key, title in [(-.18, 'accuracy', 'Test accuracy'), (.18, 'macro_f1', 'Test macro-F1')]:
        bars = ax.bar(x+offset, [r['test'][key] for r in results.values()], .36, label=title)
        ax.bar_label(bars, fmt='%.3f', padding=3)
    ax.set_xticks(x, names); ax.set_ylim(0, 1); ax.legend()
    ax.set_title('Frozen backbone + same MLP recipe | SST-5, single seed')
    ax.set_ylabel('Score (0–1)')
    fig.tight_layout(); fig.savefig(OUT/'model_comparison.png', bbox_inches='tight'); plt.close(fig)
    for name, r in results.items():
        print(f"{name}: best epoch={r['best_epoch']}, test accuracy={r['test']['accuracy']:.4f}, "
              f"macro-F1={r['test']['macro_f1']:.4f}")
    print('Charts:', OUT)


if __name__ == '__main__':
    main()
