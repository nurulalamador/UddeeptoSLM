from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "smollm2-career-merged"

SYSTEM_PROMPT = (
    "You are a helpful career guidance and skill development coach. "
    "Give clear, practical, realistic, and step-by-step advice."
)

# CPU-friendly settings
MAX_NEW_TOKENS = 120
MAX_INPUT_LENGTH = 1024
MAX_HISTORY_TURNS = 3

# Use most CPU cores but leave one core free.
CPU_THREADS = max(1, (os.cpu_count() or 2) - 1)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model folder was not found:\n{MODEL_PATH}"
        )

    print("Model path:", MODEL_PATH)
    print("CPU threads:", CPU_THREADS)

    # Must be set before CPU inference starts.
    torch.set_num_threads(CPU_THREADS)

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        use_fast=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading merged model on CPU...")

    load_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    model.eval()

    load_time = time.perf_counter() - load_start

    print(f"Model loaded successfully in {load_time:.2f} seconds.")
    print("Device: CPU")

    return model, tokenizer


def trim_history(history):
    """
    Keep the system message and only the most recent conversation turns.
    """

    max_messages = 1 + MAX_HISTORY_TURNS * 2

    if len(history) <= max_messages:
        return history

    return [history[0]] + history[-MAX_HISTORY_TURNS * 2:]


def generate_reply_streaming(
    model,
    tokenizer,
    history,
):
    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    prompt_length = inputs["input_ids"].shape[1]

    print(f"\nPrompt tokens: {prompt_length}")
    print("Assistant: ", end="", flush=True)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "repetition_penalty": 1.1,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    generation_thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs,
        daemon=True,
    )

    start_time = time.perf_counter()
    generation_thread.start()

    reply_parts = []

    for new_text in streamer:
        print(new_text, end="", flush=True)
        reply_parts.append(new_text)

    generation_thread.join()

    generation_time = time.perf_counter() - start_time
    reply = "".join(reply_parts).strip()

    print()
    print(f"\nGeneration time: {generation_time:.2f} seconds")

    return reply


def main():
    model, tokenizer = load_model()

    history = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    print("\nCareer Skill Chatbot")
    print("Type /exit to stop.")
    print("Type /clear to reset history.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nChat ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in {
            "/exit",
            "exit",
            "quit",
        }:
            print("Chat ended.")
            break

        if user_input.lower() == "/clear":
            history = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ]

            print("Conversation history cleared.")
            continue

        history.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        history = trim_history(history)

        reply = generate_reply_streaming(
            model=model,
            tokenizer=tokenizer,
            history=history,
        )

        history.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )

        history = trim_history(history)


if __name__ == "__main__":
    main()