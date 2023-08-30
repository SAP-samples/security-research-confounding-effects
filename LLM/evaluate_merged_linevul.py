import re
import argparse

import torch
import torch.nn as nn
import transformers

from datasets import load_from_disk
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, RobertaConfig


DATASETS = [
    "test",
    "perturbed-data/apply_codestyle_Chromium",
    "perturbed-data/apply_codestyle_Google",
    "perturbed-data/apply_codestyle_LLVM",
    "perturbed-data/apply_codestyle_Mozilla",
    "perturbed-data/apply_codestyle_GNU",
]

FLAW_LINES = [
    "cin>>buf;",
    "char buf[MAXSIZE];"
]

FLAW_LINE_SEP = "/~/"

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(RobertaForSequenceClassification):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
    
        
    def forward(self, input_ids=None, input_embed=None, labels=None, output_attentions=False):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, self.get_attention_scores(input_ids, attentions)
            else:
                return prob, self.get_attention_scores(input_ids, attentions)
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob
    
    def get_attention_scores(self, input_ids: torch.Tensor, att):
        att = att[0] # only first layer
        # shape: [batch_size, num_heads, seq_len, seq_len]
        att = torch.sum(att, dim=1) # => sum over all heads
        att = torch.sum(att, dim=-2) # => sum per token over all input tokens
        
        att[:, 0] = 0 # disregard start of sequence
        att[:, -1] = 0 # disregard end of sequence
        
        padding_mask = input_ids.eq(self.tokenizer.pad_token_id)
        att[padding_mask] = 0 # disregard padding
        # all that are one before padding are end-of-sequence tokens and should be disregarded as well
        att[:, :-1][padding_mask[:, 1:]] = 0

        return att

def get_scores_per_line(input_ids: torch.Tensor, att: torch.Tensor, line_seperator_ids):
    """
    input_ids: shape [batch_size, seq_len]
    att: shape [batch_size, seq_len]
    """
    # could be vectorized, but I don't want to do that
    ret = []
    for (sample, sample_att) in zip(input_ids.tolist(), att.tolist()):
        sum_score = 0
        sample_scores = []
        for token, score in zip(sample, sample_att):
            sum_score += score
            if token in line_seperator_ids:
                sample_scores.append(sum_score)
                sum_score = 0
        sample_scores.append(sum_score)
        ret.append(sample_scores)
    return ret


@torch.no_grad()
def sort_lines_batched(scores):
    def sort_lines(t):
        _, indices = torch.sort(torch.as_tensor(t), descending=True)
        return indices.tolist()
    return [sort_lines(lines) for lines in scores]


def get_flaw_indices(lines, flaw_lines):
    indices = []
    def clean(line):
        # line = re.sub("^\s", "", line)
        # line = re.sub("\s$", "", line)
        line = re.sub("\s", "", line)
        return line
    flaw_lines = [clean(flaw_line) for flaw_line in flaw_lines if len(clean(flaw_line)) != 0]
    lines = [clean(line) for line in lines]

    for i, line in enumerate(lines):
        if len(line) == 0:
            continue
        if any(line in flaw_line for flaw_line in flaw_lines) or \
            any(flaw_line in line for flaw_line in flaw_lines):
            indices.append(i)
    return indices


def min_rank_of_indices(sorted_indices, searched_indices):
    rank_mapping = {index: rank for rank, index in enumerate(sorted_indices)}
    return min(
        (rank_mapping[index] for index in searched_indices if index in rank_mapping),
        default=float("inf"),
    )

@torch.no_grad()
def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    line_seperator_ids = tokenizer.convert_tokens_to_ids(
        [token for token in tokenizer.vocab if "Ċ" in token]
    )
    
    config = RobertaConfig.from_pretrained(args.model)
    config.num_labels = 1
    model = transformers.RobertaForSequenceClassification.from_pretrained(args.model, config=config, ignore_mismatched_sizes=True)
    model = Model(model, config, tokenizer, args)

    def compute_metrics(ranks, preds):
        ranks = torch.as_tensor(ranks)
        preds = torch.as_tensor(preds).argmax(dim=-1)
        tp_ranks = ranks[preds == 1]
        ranks = ranks[~ranks.isnan()]
        tp_ranks = tp_ranks[~tp_ranks.isnan()]
        def topk_acc(k):
            return round((
                torch.sum(ranks < k) / len(ranks)
            ).item() * 100, 2)
        return {
            "Top1-Acc": topk_acc(1),
            "Top3-Acc": topk_acc(3),
            "Top5-Acc": topk_acc(5),
        }

    model.load_state_dict(torch.load(args.checkpoint), strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # prepare reference for flaw lines
    reference = load_from_disk(args.reference)
    reference.set_format(None, columns=["index", "flaw_line"])
    index_to_flaw_line = {}
    for sample in reference:
        index_to_flaw_line[sample["index"]] = sample["flaw_line"]
    
    with open("merged.txt", "r") as f:
        successful_idxs = set(map(int, f.read().split("\n")))

    for dataset_path in DATASETS:
        data = load_from_disk(dataset_path)
        data = data.filter(lambda sample: sample["label"] == 1)
        data.set_format("torch", columns=["processed_func", "label", "input_ids", "index"])

        def add_ranks(sample):
            input_ids = tokenizer(
                sample["processed_func"], truncation=True,
                padding="max_length", return_tensors="pt",
            )["input_ids"].to(device)
            pred, att = model(input_ids, output_attentions=True)
            pred = pred.cpu().tolist()
            
            line_scores = get_scores_per_line(input_ids, att, line_seperator_ids)
            sorted_lines_batch = sort_lines_batched(line_scores)
            ranks = []
            for sorted_lines, text, index in zip(sorted_lines_batch, sample["processed_func"], sample["index"]):
                index = index.item()
                if index not in index_to_flaw_line:
                    print(f"Warning: skipping index {index} not found")
                    ranks.append(float("nan"))
                    continue
                if index not in successful_idxs:
                    ranks.append(float("nan"))
                    continue
                flaws = index_to_flaw_line[index]
                if flaws is None:
                    ranks.append(float("inf"))
                    continue
                flaw_line_indices = get_flaw_indices(text.splitlines(), flaws.split(FLAW_LINE_SEP))
                ranks.append(
                    float(min_rank_of_indices(sorted_indices=sorted_lines, searched_indices=flaw_line_indices))
                )
            
            return {"rank": ranks, "pred": pred}

        with torch.cuda.amp.autocast():
            data = data.filter(lambda row: int(row["index"]) in successful_idxs).map(add_ranks, batched=True, batch_size=8)
        print(f"Evaluating {dataset_path}")
        print(compute_metrics(data["rank"], data["pred"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="microsoft/codebert-base")
    parser.add_argument("--checkpoint", type=str, default="model_standard/linevul/12heads_linevul_model.bin")
    parser.add_argument("--dataset", type=str, default="test")
    parser.add_argument("--reference", type=str, default="test")

    args = parser.parse_args()

    main(args)