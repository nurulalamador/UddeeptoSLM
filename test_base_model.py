
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

PROJECT_DIR = Path(
    "/kaggle/working/career_slm_project"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "outputs"
    / "test-results"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "base_model_test_results.jsonl"
)

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


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True
    )

    model.eval()

    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    question: str
) -> dict:

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

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    prompt_tokens = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    torch.cuda.synchronize()
    generation_time = time.perf_counter() - start_time

    generated_ids = output[0][prompt_tokens:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    generated_tokens = len(generated_ids)

    speed = (
        generated_tokens / generation_time
        if generation_time > 0
        else 0
    )

    return {
        "question": question,
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "generation_time_seconds": round(
            generation_time,
            4
        ),
        "tokens_per_second": round(
            speed,
            4
        )
    }


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model, tokenizer = load_model()

    results = []

    for index, question in enumerate(
        TEST_QUESTIONS,
        start=1
    ):
        print("\n" + "=" * 80)
        print(f"BASE MODEL TEST {index}")
        print("=" * 80)

        result = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question
        )

        results.append(result)

        print("\nQuestion:")
        print(question)

        print("\nBase model answer:")
        print(result["answer"])

        print(
            "\nGeneration time:",
            result["generation_time_seconds"],
            "seconds"
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

    print("\nBase model results saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()
