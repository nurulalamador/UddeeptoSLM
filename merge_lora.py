from pathlib import Path
import torch

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

PROJECT_DIR = Path(__file__).resolve().parent

ADAPTER_DIR = PROJECT_DIR / "smollm2-career-lora-final"
MERGED_DIR = PROJECT_DIR / "smollm2-career-merged"


def main():
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Adapter folder not found:\n{ADAPTER_DIR}"
        )

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]

    for filename in required_files:
        file_path = ADAPTER_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required adapter file missing:\n{file_path}"
            )

    MERGED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model on CPU...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    print("Loading LoRA adapter...")

    peft_model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_DIR),
        is_trainable=False
    )

    print("Merging LoRA weights with base model...")

    merged_model = peft_model.merge_and_unload()

    print("Saving merged model...")

    merged_model.save_pretrained(
        MERGED_DIR,
        safe_serialization=True,
        max_shard_size="2GB"
    )

    tokenizer.save_pretrained(
        MERGED_DIR
    )

    print("\nMerge completed successfully.")
    print("Merged model saved to:")
    print(MERGED_DIR)


if __name__ == "__main__":
    main()