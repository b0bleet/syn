---
base_model: Qwen/Qwen3-8B-Base
tags:
  - option-scoring
  - experimental
---

# Syn-0.1

Syn-0.1 is an early experimental release of [Syn](https://github.com/b0bleet/syn), the option-scoring service behind [sifty.dev](https://sifty.dev). This repository contains data, benchmark results, logs, and checkpoints. It is an artifact collection, not a single model loadable from the repository root.

The pointer run adapts [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base); its weights were not trained from scratch. Results are experiment-specific. Check the run configuration and evaluation before using a checkpoint.

| Path | Contents |
| --- | --- |
| `data/` | Labeled rows for training and evaluation. |
| `pointer-data/` | Pointer training rows and data provenance. |
| `pointers/` | Adapted backbone and pointer checkpoints. |
| `jevbench/` | Benchmark artifacts. |
| `logs/` | Training logs. |

The [Syn README](https://github.com/b0bleet/syn#readme) explains the scoring API and training methods. The [RunPod guide](https://github.com/b0bleet/syn/blob/main/deploy/runpod/README.md) explains how to train with this repository as the artifact store. For upload or training jobs that publish artifacts, use an HF_TOKEN with write access to `siftylabs/Syn-0.1`; public downloads do not need a token.

To serve an available pointer run, set `SYN_MODEL=hf://siftylabs/Syn-0.1/pointers/<model>/<run>/backbone`, `SYN_READOUT=pointer`, and `SYN_POINTER_PATH=hf://siftylabs/Syn-0.1/pointers/<model>/<run>/pointer.safetensors`. Replace `<model>` and `<run>` with a directory listed in this repository.