
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Configuration
# ============================================================

BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

PROJECT_DIR = Path("/kaggle/working/career_slm_project")

ADAPTER_PATH = (
    PROJECT_DIR
    / "outputs"
    / "smollm2-career-lora-final"
)

RESULTS_DIR = PROJECT_DIR / "outputs" / "test-results"
RESULTS_FILE = RESULTS_DIR / "fine_tuned_test_results.jsonl"

SYSTEM_PROMPT = (
    "You are a helpful career guidance and skill development coach. "
    "Give clear, practical, realistic, and step-by-step advice."
)

MAX_NEW_TOKENS = 400
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1


TEST_QUESTIONS = [
    "I want to become a data analyst. What skills should I learn?",

    "How can I improve my communication skills in 30 days?",

    "I am a beginner and want to start freelancing. "
    "Which skill should I choose?",

    "Give me a practical three-month career development roadmap.",

    "How should I prepare for a job interview?"
]


# ============================================================
# Environment check
# ============================================================

def check_environment() -> torch.device:
    print("=" * 80)
    print("ENVIRONMENT CHECK")
    print("=" * 80)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU VRAM:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2
            ),
            "GB"
        )
    else:
        device = torch.device("cpu")
        print("GPU was not detected. CPU inference will be used.")

    print("Selected device:", device)
    print()

    return device


# ============================================================
# Path validation
# ============================================================

def validate_adapter_path() -> None:
    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"LoRA adapter folder was not found:\n{ADAPTER_PATH}"
        )

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors"
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (ADAPTER_PATH / filename).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "The adapter folder is incomplete. Missing files: "
            + ", ".join(missing_files)
        )

    print("Adapter folder found:")
    print(ADAPTER_PATH)
    print()


# ============================================================
# Model loading
# ============================================================

def load_model_and_tokenizer(
    device: torch.device
):
    print("=" * 80)
    print("LOADING TOKENIZER")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print("Tokenizer loaded successfully.")
    print()

    print("=" * 80)
    print("LOADING BASE MODEL")
    print("=" * 80)

    if device.type == "cuda":
        model_dtype = torch.float16

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=model_dtype,
            device_map={"": 0},
            low_cpu_mem_usage=True
        )
    else:
        model_dtype = torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=model_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True
        )

    base_model.config.use_cache = True

    print("Base model loaded successfully.")
    print()

    print("=" * 80)
    print("LOADING LORA ADAPTER")
    print("=" * 80)

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
        is_trainable=False
    )

    model.eval()

    print("LoRA adapter loaded successfully.")
    print()

    return model, tokenizer


# ============================================================
# Prompt creation
# ============================================================

def build_prompt(
    tokenizer,
    question: str
) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt


# ============================================================
# Response generation
# ============================================================

def generate_response(
    model,
    tokenizer,
    question: str,
    device: torch.device
) -> dict:
    prompt = build_prompt(
        tokenizer=tokenizer,
        question=question
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    prompt_token_count = inputs["input_ids"].shape[1]

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    generation_time = time.perf_counter() - start_time

    generated_token_ids = outputs[0][prompt_token_count:]

    answer = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True
    ).strip()

    generated_token_count = len(generated_token_ids)

    tokens_per_second = (
        generated_token_count / generation_time
        if generation_time > 0
        else 0.0
    )

    return {
        "question": question,
        "answer": answer,
        "prompt_tokens": prompt_token_count,
        "generated_tokens": generated_token_count,
        "generation_time_seconds": round(
            generation_time,
            4
        ),
        "tokens_per_second": round(
            tokens_per_second,
            4
        )
    }


# ============================================================
# Save results
# ============================================================

def save_results(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:
        for result in results:
            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + "\n"
            )

    print()
    print("Test results saved to:")
    print(RESULTS_FILE)


# ============================================================
# Main
# ============================================================

def main() -> None:
    device = check_environment()

    validate_adapter_path()

    model, tokenizer = load_model_and_tokenizer(
        device=device
    )

    all_results = []

    print("=" * 80)
    print("STARTING FINE-TUNED MODEL TEST")
    print("=" * 80)

    for index, question in enumerate(
        TEST_QUESTIONS,
        start=1
    ):
        print()
        print("=" * 80)
        print(f"TEST PROMPT {index}")
        print("=" * 80)

        print("Question:")
        print(question)

        result = generate_response(
            model=model,
            tokenizer=tokenizer,
            question=question,
            device=device
        )

        all_results.append(result)

        print()
        print("Fine-tuned model answer:")
        print(result["answer"])

        print()
        print(
            "Generated tokens:",
            result["generated_tokens"]
        )

        print(
            "Generation time:",
            result["generation_time_seconds"],
            "seconds"
        )

        print(
            "Generation speed:",
            result["tokens_per_second"],
            "tokens/second"
        )

    save_results(all_results)

    print()
    print("=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
