import os
import argparse

from tqdm import tqdm

from datasets import Dataset
from transformers import AutoTokenizer


def main(args):
    path = args.path.replace("\\", "/")
    files = list(filter(lambda s: s.endswith(".c"), os.listdir(path)))

    save_path = os.path.join("data/hf", os.path.split(path.rstrip("/"))[1])
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    data = {"processed_func": [], "target": [], "index": [],}
    for file in tqdm(files):
        path = os.path.join(args.path, file)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        label = int(file.split(".")[0].split("_")[-1])
        data["processed_func"].append(content)
        data["target"].append(label)
        data["index"].append(int(file.split("_")[0]))
    
    d = Dataset.from_dict(data)
    d = d.map(lambda row: tokenizer(
        text=row["processed_func"],
        padding="max_length",
        truncation=True,
    ), batched=True).rename_column("target", "label")
    d.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    d.save_to_disk(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str)
    parser.add_argument("--model", type=str, default="Salesforce/codet5p-220m")

    args = parser.parse_args()

    main(args)