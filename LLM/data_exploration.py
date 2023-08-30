import os
import argparse

import transformers
import datasets


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Salesforce/codet5p-220m")
    parser.add_argument("dataset_path", type=str)

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    d = datasets.Dataset.from_csv(args.dataset_path)

    d = d.map(lambda row: tokenizer(
        text=row["processed_func"],
        padding="max_length",
        truncation=True,
    ), batched=True).rename_column("target", "label")
    d.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    d.save_to_disk(os.path.split(args.dataset_path)[1].split(".")[0])