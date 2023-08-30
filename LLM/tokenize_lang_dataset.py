import os
import argparse

import transformers
from datasets import load_from_disk

from lang_processors import CppProcessor

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Salesforce/codet5p-220m")
    parser.add_argument("dataset_path", type=str)

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    d = load_from_disk(args.dataset_path)
    proc = CppProcessor()

    def tokenize(text):
        return " ".join(proc.tokenize_code(text))

    prev_columns = d.column_names
    prev_columns.remove("label")
    d = d.map(lambda row: tokenizer(
        text=list(map(tokenize, row["processed_func"])),
        padding="max_length",
        truncation=True,
    ), batched=True, remove_columns=prev_columns)
    d.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    old_dir, old_tail = os.path.split(args.dataset_path.rstrip("/"))
    new_path = os.path.join(old_dir, old_tail+"-tokenized-linevul")
    d.save_to_disk(new_path)