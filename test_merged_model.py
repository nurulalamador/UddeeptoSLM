
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


MODEL_PATH = Path(
    "/kaggle/working/career_slm_project/"
    "outputs/smollm2-career-merged"
)

SYSTEM_PROMPT = (
    "You are a helpful career guidance and "
    "skill development coach. "
    "Give clear, practical, realistic, "
    "and step-by-step advice."
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True
)

model.eval()

question = (
    "I want to become a data analyst. "
    "What skills should I learn?"
)

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
    return_tensors="pt"
).to("cuda")

prompt_length = (
    inputs["input_ids"].shape[1]
)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

generated_ids = (
    output[0][prompt_length:]
)

answer = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True
)

print(answer)
