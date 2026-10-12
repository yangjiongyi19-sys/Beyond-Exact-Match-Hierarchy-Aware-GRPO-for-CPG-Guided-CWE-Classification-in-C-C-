"""Assistant-only causal SFT over verified multi-turn tool traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _render_and_mask(tokenizer: Any, messages: list[dict[str, Any]], max_length: int) -> dict[str, list[int]]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    spans: list[tuple[int, int]] = []
    assistant_end_positions: list[int] = []
    cursor = 0
    for message in messages:
        role = str(message.get("role", ""))
        marker = f"<|im_start|>{role}"
        marker_at = rendered.find(marker, cursor)
        if marker_at < 0:
            raise ValueError(f"Could not locate rendered chat role {role!r}; check the tokenizer template")
        content_start = rendered.find("\n", marker_at + len(marker))
        if content_start < 0:
            raise ValueError(f"Could not locate message content for role {role!r}")
        content_start += 1
        message_end = rendered.find("<|im_end|>", content_start)
        if message_end < 0:
            message_end = rendered.find("<|im_start|>", content_start)
        if message_end < 0:
            message_end = len(rendered)
        content = str(message.get("content", ""))
        expected_at = rendered.find(content, content_start, message_end) if content else content_start
        if expected_at < 0:
            raise ValueError("Rendered message content does not match its source message")
        if role == "assistant" and content:
            spans.append((expected_at, expected_at + len(content)))
        if role == "assistant" and rendered.startswith("<|im_end|>", message_end):
            assistant_end_positions.append(message_end)
        cursor = message_end + len("<|im_end|>") if rendered.startswith("<|im_end|>", message_end) else message_end

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = list(encoded["input_ids"])
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer did not return offset_mapping; use a fast Qwen tokenizer")
    labels = []
    for token_id, (start, end) in zip(input_ids, offsets):
        supervised = end > start and any(start < span_end and end > span_start for span_start, span_end in spans)
        labels.append(token_id if supervised else -100)
    # Fast-tokenizer offsets are commonly (0, 0) for special tokens. Explicitly
    # supervise assistant turn terminators so SFT learns when a tool turn ends.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for char_position in assistant_end_positions:
        token_position = len(tokenizer(rendered[:char_position], add_special_tokens=False)["input_ids"])
        if token_position < len(input_ids) and input_ids[token_position] == im_end_id:
            labels[token_position] = input_ids[token_position]
    if not any(label != -100 for label in labels):
        raise ValueError("No assistant tokens remain after truncation")
    return {"input_ids": input_ids, "labels": labels}


class _TraceDataset:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class _AssistantOnlyCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        input_ids, labels, attention_mask = [], [], []
        for feature in features:
            pad = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad)
            labels.append(feature["labels"] + [-100] * pad)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def train(
    model_name_or_path: str,
    traces_path: str | Path,
    output_dir: str | Path,
    *,
    max_length: int = 8192,
    epochs: float = 3.0,
    learning_rate: float = 2e-5,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    save_steps: int = 500,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    rows = []
    with Path(traces_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            messages = item.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"SFT trace line {line_number} is missing messages")
            rows.append(_render_and_mask(tokenizer, messages, max_length))
    if not rows:
        raise ValueError("No SFT traces were loaded")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=True,
        gradient_checkpointing=True,
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"]},
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_TraceDataset(rows),
        data_collator=_AssistantOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=500)
    args = parser.parse_args()
    train(
        args.model,
        args.traces,
        args.out,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
    )


if __name__ == "__main__":
    main()

