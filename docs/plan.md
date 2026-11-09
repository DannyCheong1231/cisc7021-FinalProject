# SST-5 frozen-backbone classifier

Goal: download pretrained Llama-style weights, freeze the backbone, extract the
last layer at the last non-padding input token, and train only a five-class head.
The PDF is project context; its RTO/baseline tasks are outside this request.

Implementation choices: Python/PyTorch/Transformers; direct pinned Stanford
fiveclass trees, sentence-root labels only; no generated tokens or chat prompt.
Support the course 42M checkpoint and public TinyLlama through --model.
Default MLP: LayerNorm -> Linear(128) -> ReLU -> Dropout -> Linear(5).
Cache frozen features once; select epoch by validation macro-F1, test once.
Internal IDs 0..4 correspond to display levels 1..5, negative to positive.

1. Test root-label parsing, left/right padding, frozen backbone and head updates.
   Implement only the data/model helpers needed to pass these checks.
2. Implement training, checkpoint reload and prediction; verify with a tiny random
   Llama locally, then the downloaded pretrained checkpoint and real SST splits.
3. Run full head training; save revision IDs, hyperparameters, metrics, confusion
   matrix and predictions. Document setup and commands, plus measured limitations.

Success: backbone has no gradients; head weights change; padding cannot select a
PAD hidden state; split boundaries remain intact; predictions reload from disk
and map to the requested five labels. No promised accuracy threshold.
