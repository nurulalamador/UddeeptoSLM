# combine_career_datasets.py

from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import re


SYSTEM_PROMPT = (
    "You are a helpful career guidance and skill development coach. "
    "Give clear, practical, step-by-step advice."
)


def clean_text(text):
    """Basic text cleaning."""
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_chat_text(instruction, response):
    """
    Convert instruction-response pair into a simple chat-style text.
    This text field will be useful for SFTTrainer.
    """
    instruction = clean_text(instruction)
    response = clean_text(response)

    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{instruction}\n"
        f"<|assistant|>\n{response}"
    )


def normalize_pradeep():
    """
    Dataset: Pradeep016/career-guidance-qa-dataset
    Expected columns: Role, Question, Answer
    Final columns: instruction, response, source, text
    """
    print("Loading Pradeep016/career-guidance-qa-dataset...")

    ds = load_dataset("Pradeep016/career-guidance-qa-dataset")

    # Usually has only train split
    split_name = list(ds.keys())[0]
    df = ds[split_name].to_pandas()

    print("Pradeep columns:", df.columns.tolist())

    normalized_rows = []

    for _, row in df.iterrows():
        role = clean_text(row.get("Role", ""))
        question = clean_text(row.get("Question", ""))
        answer = clean_text(row.get("Answer", ""))

        if not question or not answer:
            continue

        if role:
            instruction = f"For the career role '{role}', answer this question: {question}"
        else:
            instruction = question

        response = answer

        normalized_rows.append({
            "instruction": instruction,
            "response": response,
            "source": "Pradeep016/career-guidance-qa-dataset",
            "text": make_chat_text(instruction, response)
        })

    return Dataset.from_pandas(pd.DataFrame(normalized_rows), preserve_index=False)


def normalize_vamshi():
    """
    Dataset: VamshiSurya/Career-Guidance
    Expected columns: instruction, response
    Final columns: instruction, response, source, text
    """
    print("Loading VamshiSurya/Career-Guidance...")

    ds = load_dataset("VamshiSurya/Career-Guidance")

    split_name = list(ds.keys())[0]
    df = ds[split_name].to_pandas()

    print("Vamshi columns:", df.columns.tolist())

    normalized_rows = []

    for _, row in df.iterrows():
        instruction = clean_text(row.get("instruction", ""))
        response = clean_text(row.get("response", ""))

        if not instruction or not response:
            continue

        normalized_rows.append({
            "instruction": instruction,
            "response": response,
            "source": "VamshiSurya/Career-Guidance",
            "text": make_chat_text(instruction, response)
        })

    return Dataset.from_pandas(pd.DataFrame(normalized_rows), preserve_index=False)


def normalize_dhanunjaiah():
    """
    Dataset: Dhanunjaiah/llama_career_guidance_dataset
    Expected columns: Instruction, Input, Response, Prompt
    Final columns: instruction, response, source, text
    """
    print("Loading Dhanunjaiah/llama_career_guidance_dataset...")

    ds = load_dataset("Dhanunjaiah/llama_career_guidance_dataset")

    split_name = list(ds.keys())[0]
    df = ds[split_name].to_pandas()

    print("Dhanunjaiah columns:", df.columns.tolist())

    normalized_rows = []

    for _, row in df.iterrows():
        instruction_part = clean_text(row.get("Instruction", ""))
        input_part = clean_text(row.get("Input", ""))
        response = clean_text(row.get("Response", ""))

        if not response:
            continue

        if instruction_part and input_part:
            instruction = f"{instruction_part}\n\nUser profile/context: {input_part}"
        elif instruction_part:
            instruction = instruction_part
        elif input_part:
            instruction = input_part
        else:
            continue

        normalized_rows.append({
            "instruction": instruction,
            "response": response,
            "source": "Dhanunjaiah/llama_career_guidance_dataset",
            "text": make_chat_text(instruction, response)
        })

    return Dataset.from_pandas(pd.DataFrame(normalized_rows), preserve_index=False)


def remove_duplicates(dataset):
    """
    Remove exact duplicate instruction-response pairs.
    """
    df = dataset.to_pandas()

    before = len(df)

    df["instruction"] = df["instruction"].apply(clean_text)
    df["response"] = df["response"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)

    df = df.drop_duplicates(subset=["instruction", "response"])
    df = df.reset_index(drop=True)

    after = len(df)

    print(f"Removed duplicates: {before - after}")
    print(f"Final rows after duplicate removal: {after}")

    return Dataset.from_pandas(df, preserve_index=False)


def main():
    pradeep_ds = normalize_pradeep()
    vamshi_ds = normalize_vamshi()
    dhanunjaiah_ds = normalize_dhanunjaiah()

    print("\nDataset sizes before combine:")
    print("Pradeep:", len(pradeep_ds))
    print("Vamshi:", len(vamshi_ds))
    print("Dhanunjaiah:", len(dhanunjaiah_ds))

    combined = concatenate_datasets([
        pradeep_ds,
        vamshi_ds,
        dhanunjaiah_ds
    ])

    print("\nCombined size before duplicate removal:", len(combined))

    combined = remove_duplicates(combined)

    # Shuffle dataset
    combined = combined.shuffle(seed=42)

    # Train-validation split
    split_dataset = combined.train_test_split(test_size=0.1, seed=42)

    train_ds = split_dataset["train"]
    valid_ds = split_dataset["test"]

    print("\nFinal split:")
    print("Train:", len(train_ds))
    print("Validation:", len(valid_ds))

    # Save as JSONL
    combined.to_json("data/combined_career_dataset.jsonl", orient="records", lines=True, force_ascii=False)
    train_ds.to_json("data/train.jsonl", orient="records", lines=True, force_ascii=False)
    valid_ds.to_json("data/valid.jsonl", orient="records", lines=True, force_ascii=False)

    print("\nSaved files:")
    print("- data/combined_career_dataset.jsonl")
    print("- data/train.jsonl")
    print("- data/valid.jsonl")

    # Show one sample
    print("\nSample normalized row:")
    print(combined[0])


if __name__ == "__main__":
    main()