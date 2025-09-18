LLM Activation Analysis

This folder contains a script to load a Hugging Face LLM (default: Qwen/Qwen3-8B), run it on a subset of a text dataset, collect per-layer activations excluding padding, and build histograms of activation values.

Outputs include per-layer histogram PNGs and a histograms.npz (counts and bins).

Requirements
- Python 3.10+
- Packages: transformers, datasets, torch, matplotlib, numpy

Install (example using pip):
- pip install --upgrade transformers datasets torch matplotlib numpy

Usage examples
1) Default settings (ag_news, first 128 samples), with Qwen3-8B:
- python llm_analysis/analyze_activations.py --model Qwen/Qwen3-8B --num_samples 64 --batch_size 1

2) Faster/CPU-friendly smoke test:
- python llm_analysis/analyze_activations.py --model Qwen/Qwen3-8B --num_samples 8 --batch_size 1 --max_length 256

3) Different dataset (e.g., wikitext-2-raw-v1):
- python llm_analysis/analyze_activations.py --model Qwen/Qwen3-8B --dataset wikitext --split wikitext-2-raw-v1 --text_column text --num_samples 64

Notes
- The script requests output_hidden_states from the model and aggregates histograms per layer (including the embeddings layer at index 0).
- Padding positions are excluded using the attention_mask.
- Histograms are accumulated using fixed bins in the range [min_val, max_val] (default [-10, 10]). Values outside this range are clipped.
- On GPUs, device_map="auto" and bfloat16 will be used when supported; otherwise it falls back to float16 or CPU float32.
- Some models may not define a pad token. The script uses the eos token as pad if available, otherwise it creates a <|pad|> token and resizes embeddings.

Outputs
- llm_analysis/outputs/<model_tag>/
  - histograms.npz (counts [num_layers, num_bins-1], bins [num_bins])
  - layer_XX.png (bar plots of activation histograms per layer)
  - run_info.txt (run configuration)
