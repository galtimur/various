import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import fire


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}


def get_device_kwargs(prefer_bf16: bool = True):
    # Choose dtype based on hardware
    torch_dtype = None
    if torch.cuda.is_available():
        try:
            if prefer_bf16 and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
        except Exception:
            torch_dtype = torch.float16
    else:
        # CPU fallback
        torch_dtype = torch.float32
    return {
        "device_map": "auto" if torch.cuda.is_available() else None,
        "torch_dtype": torch_dtype,
    }


def ensure_pad_token(tokenizer, model):
    # Many LLMs don't have an explicit pad token; use eos as pad if needed
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Create a new pad token if absolutely necessary
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def prepare_data(dataset_name: str, split: str, text_column: str | None, num_samples: int,
                 tokenizer: AutoTokenizer, max_length: int, batch_size: int) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    ds = load_dataset(dataset_name, split=split)
    if text_column is None:
        # Try to auto-detect a text column
        for candidate in ["text", "sentence", "content", "article", "review", "document"]:
            if candidate in ds.column_names:
                text_column = candidate
                break
        if text_column is None:
            # Fall back to the first string column
            for c in ds.column_names:
                if isinstance(ds[0][c], str):
                    text_column = c
                    break
    if text_column is None:
        raise ValueError("Could not determine text column. Specify --text_column explicitly.")

    ds = ds.select(range(min(num_samples, len(ds))))

    def tok(batch):
        return tokenizer(batch[text_column], padding=False, truncation=True, max_length=max_length)

    tokenized = ds.map(tok, batched=True, remove_columns=[c for c in ds.column_names if c != text_column])

    # Create batches with dynamic padding per batch
    batches: List[Dict[str, torch.Tensor]] = []
    cur = []
    for ex in tokenized:
        cur.append(ex)
        if len(cur) >= batch_size:
            batches.append(collate(cur, tokenizer))
            cur = []
    if cur:
        batches.append(collate(cur, tokenizer))
    total = len(ds)
    return batches, total


def collate(examples: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    # Pad to the longest in the batch
    max_len = max(len(e["input_ids"]) for e in examples)
    input_ids = []
    attn_mask = []
    for e in examples:
        ids = e["input_ids"]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = ids + [tokenizer.pad_token_id] * pad_len
        input_ids.append(ids)
        attn_mask.append([1] * (max_len - pad_len) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask, dtype=torch.long)
    }


def init_hist(num_layers: int, bins: np.ndarray) -> List[np.ndarray]:
    return [np.zeros(len(bins) - 1, dtype=np.int64) for _ in range(num_layers)]


def update_hist(layer_hists: List[np.ndarray], bins: np.ndarray, hidden_states: Tuple[torch.Tensor, ...],
                attention_mask: torch.Tensor, clip_range: Tuple[float, float] | None = (-10.0, 10.0)):
    # hidden_states is a tuple: [embeddings, layer1, layer2, ..., layerN]
    # We will build histogram for each layer including the embedding layer for completeness.
    # Exclude padding positions using attention_mask.
    bsz, seqlen = attention_mask.shape
    mask = attention_mask.bool().view(bsz, seqlen, 1)  # broadcast over hidden size

    for i, hs in enumerate(hidden_states):
        # hs: [B, S, H]
        if hs is None:
            continue
        x = hs.masked_select(mask)  # shape: (num_nonpad * H,)
        if x.numel() == 0:
            continue
        if clip_range is not None:
            x = x.clamp(min=clip_range[0], max=clip_range[1])
        # Move to CPU numpy
        x_np = x.detach().float().cpu().numpy()
        counts, _ = np.histogram(x_np, bins=bins)
        layer_hists[i] += counts


def plot_histograms(layer_hists: List[np.ndarray], bins: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save raw counts and bins
    # np.savez(out_dir / "histograms.npz", counts=np.stack(layer_hists, axis=0), bins=bins)

    # Plot per-layer histograms
    centers = (bins[:-1] + bins[1:]) / 2
    for idx, counts in enumerate(layer_hists):
        plt.figure(figsize=(8, 4))
        plt.bar(centers, counts, width=(bins[1] - bins[0]), align='center')
        plt.title(f"Layer {idx} activation histogram")
        plt.xlabel("Activation value")
        plt.ylabel("Count")
        plt.tight_layout()
        # plt.savefig(out_dir / f"layer_{idx:02d}.png", dpi=150)
        plt.show()


def analyze_activations(
    model: str = "Qwen/Qwen3-8B",
    dataset: str = "ag_news",
    split: str = "train",
    text_column: str | None = None,
    num_samples: int = 128,
    max_length: int = 512,
    batch_size: int = 1,
    bins: int = 201,
    min_val: float = -10.0,
    max_val: float = 10.0,
    output_dir: str | None = None,
    prefer_bf16: bool = True,
):
    """Analyze per-layer activation histograms for an HF LLM.

    Parameters mirror the previous argparse options. Use with Python Fire, e.g.:
      python llm_analysis/analyze_activations.py --model Qwen/Qwen3-8B --num_samples 64
    """

    # Prepare output directory
    model_tag = model.replace("/", "__")
    out_dir = Path(output_dir) if output_dir else Path("llm_analysis") / "outputs" / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device_kwargs = get_device_kwargs(prefer_bf16=prefer_bf16)

    print(f"Loading tokenizer: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    print(f"Loading model: {model}")
    model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
        **device_kwargs,
    )
    model = model.cuda()

    ensure_pad_token(tokenizer, model)

    print(f"Preparing dataset: {dataset} [{split}], samples={num_samples}")
    batches, total = prepare_data(
        dataset_name=dataset,
        split=split,
        text_column=text_column,
        num_samples=num_samples,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )

    # Warm up one forward to determine number of layers by requesting hidden states
    print("Warming up to determine layer count...")
    model.eval()
    with torch.no_grad():
        sample_batch = batches[0]
        sample_batch_dev = {k: (v.to(model.device) if hasattr(model, "device") and device_kwargs["device_map"] is None else v) for k, v in sample_batch.items()}
        outputs = model(**sample_batch_dev, output_hidden_states=True, use_cache=False, return_dict=True)
        num_layers = len(outputs.hidden_states)  # includes embedding layer
    print(f"Model provides {num_layers} hidden_states entries (including embeddings layer).")

    bin_edges = np.linspace(min_val, max_val, bins)
    layer_hists = init_hist(num_layers, bin_edges)

    processed = 0
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}...")
        # Move batch to devices if model is on a single device; for device_map=auto, pass CPU tensors.
        if device_kwargs["device_map"] is None:
            batch_dev = {k: v.to(model.device) for k, v in batch.items()}
        else:
            batch_dev = batch  # let HF handle device placement

        with torch.no_grad():
            out = model(**batch_dev, output_hidden_states=True, use_cache=False, return_dict=True)
            hs = out.hidden_states  # tuple of tensors
            attn = batch["attention_mask"]
            if device_kwargs["device_map"] is None:
                attn = attn.to(model.device)
            update_hist(layer_hists, bin_edges, hs, attn, clip_range=(min_val, max_val))
        processed += batch["input_ids"].shape[0]

    print(f"Processed {processed} samples. Saving outputs to {out_dir}...")
    plot_histograms(layer_hists, bin_edges, out_dir)

    # Also save a quick README-like info file
    with open(out_dir / "run_info.txt", "w", encoding="utf-8") as f:
        f.write(
            "\n".join([
                f"model={model}",
                f"dataset={dataset}",
                f"split={split}",
                f"num_samples={num_samples}",
                f"max_length={max_length}",
                f"batch_size={batch_size}",
                f"bins={bins}",
                f"range=[{min_val}, {max_val}]",
            ])
        )

    print("Done.")


if __name__ == "__main__":
    fire.Fire(analyze_activations)
