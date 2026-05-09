# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OmniVoice is a massively multilingual zero-shot text-to-speech (TTS) model supporting 600+ languages, built on a diffusion language model architecture. It supports voice cloning, voice design (via speaker attributes), and auto voice modes.

## Common Commands

```bash
# Install in editable mode
pip install -e .

# Using uv
uv sync                    # Core dependencies
uv sync --extra eval       # Include evaluation tools

# Run inference (single item)
uv run omnivoice-infer --help

# Run batch folder inference (reads .txt, outputs .wav)
uv run omnivoice-infer-folder --help

# Run batch inference (multi-GPU)
uv run omnivoice-infer-batch --help

# Launch Gradio demo
uv run omnivoice-demo

# Training
uv run omnivoice-train --config examples/config/train_config_finetune.json

# Data preparation scripts
uv run omnivoice-extract-audio-tokens ...
uv run omnivoice-jsonl-to-webdataset ...
```

There is no test suite or linting configuration in this repo.

## Architecture

OmniVoice uses a **diffusion LM** approach (iterative masked unmasking) rather than autoregressive decoding. Audio is represented as 8 parallel codebook layers from the Higgs Audio V2 tokenizer.

### Core Model (`omnivoice/models/omnivoice.py`)

- **`OmniVoice`**: Main `PreTrainedModel` wrapping a Qwen3-0.6B LLM backbone with added audio embedding layers and 8 audio prediction heads (one per codebook).
- **`OmniVoiceConfig`**: `PretrainedConfig` subclass — controls audio vocab size, codebook loss weights, attention implementation.
- **`OmniVoiceGenerationConfig`**: Dataclass controlling inference — `num_step` (diffusion steps, default 32), `guidance_scale`, `position_temperature`, `class_temperature`, `layer_penalty_factor`.
- **`GenerationTask`** / **`VoiceClonePrompt`**: Input dataclasses for the generation pipeline.

### Generation Pipeline (`OmniVoice.generate()`)

1. Text → HuggingFace tokenizer
2. Optional `ref_audio` → Higgs Audio V2 tokenizer → speaker embedding (voice cloning)
3. Optional `instruct` → speaker attributes → encoded prompt (voice design)
4. Rule-based duration estimation (`omnivoice/utils/duration.py`)
5. Auto-chunk long texts (> ~30s estimated duration) via `omnivoice/utils/text.py`
6. Iterative diffusion decoding (masked language modeling over audio tokens)
7. Post-processing: silence removal, cross-fading, format conversion (`omnivoice/utils/audio.py`)

### Training Pipeline (`omnivoice/training/`)

- **`OmniTrainer`** (`trainer.py`): Wraps HuggingFace Accelerate for distributed training, bf16, DeepSpeed, gradient accumulation, checkpoint save/resume.
- **`TrainingConfig`** (`config.py`): All hyperparameters and path configs as a dataclass; loaded from JSON.
- **`build_model`** / **`build_dataloader`** (`builder.py`): Factory functions wired to the CLI.
- Data: `PackingIterableDataset` + `PackingDataCollator` handle dynamic packing of variable-length sequences from WebDataset shards or JSONL manifests.

### Key Utilities

| File | Purpose |
|------|---------|
| `utils/lang_map.py` | 600+ language ID/name bidirectional mappings |
| `utils/text.py` | Punctuation-aware text chunking, normalization, pinyin/CMU phoneme insertion |
| `utils/audio.py` | Load, resample, silence removal, cross-fade, long-audio chunking |
| `utils/voice_design.py` | Speaker attribute validation and EN↔Chinese translation |
| `utils/duration.py` | Rule-based per-language duration estimation |

### Data Format

Training data can be either **WebDataset shards** (`.tar`) or **JSONL manifests**. Each sample requires: audio tokens, text tokens, language ID, and optionally a speaker prompt. See `docs/data_preparation.md` for the schema.

### CLI Entry Points

| Command | Module | Use |
|---------|--------|-----|
| `omnivoice-infer` | `omnivoice.cli.infer` | Single-item inference |
| `omnivoice-infer-folder` | `omnivoice.cli.infer_folder` | Batch folder inference (.txt → .wav) |
| `omnivoice-infer-batch` | `omnivoice.cli.infer_batch` | Multi-GPU batch via `ProcessPoolExecutor` |
| `omnivoice-demo` | `omnivoice.cli.demo` | Gradio web UI |

### Non-Verbal Symbols & Pronunciation

Text supports special tokens like `[laughter]`, `[sigh]`, pinyin annotations for Chinese (e.g., `汉[han4]字[zi4]`), and CMU phoneme annotations for English. These are handled during text tokenization in the model's `_prepare_inputs` method.
