import re
import sys
sys.setrecursionlimit(15000)
import argparse

import torch
import torch.nn as nn
import transformers

from datasets import load_from_disk
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, RobertaConfig
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, balanced_accuracy_score

from lang_processors import CppProcessor


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


@torch.no_grad()
def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    
    config = RobertaConfig.from_pretrained(args.model)
    config.num_labels = 1
    model = transformers.RobertaForSequenceClassification.from_pretrained(args.model, config=config, ignore_mismatched_sizes=True)
    model = Model(model, config, tokenizer, args)

    def compute_metrics(pred, true):
        predicted = torch.as_tensor(pred).argmax(-1)
        return {
            "MCC": matthews_corrcoef(true, predicted),
            "F1": f1_score(true, predicted, average='macro'),
            "Acc": accuracy_score(true, predicted),
            "BAcc": balanced_accuracy_score(true, predicted),
        }

    model.load_state_dict(torch.load(args.checkpoint), strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data = load_from_disk("reveal")
    data.set_format("torch", columns=["processed_func", "label"])

    proc = CppProcessor()

    def tokenize(text):
        return " ".join(proc.tokenize_code(text))

    def add_predictions(sample):
        input_ids = tokenizer(
            list(map(tokenize, sample["processed_func"])), truncation=True,
            padding="max_length", return_tensors="pt",
        )["input_ids"].to(device)
        pred = model(input_ids).cpu().tolist()
        true = sample["label"]
        return {"pred": pred, "true": true}

    with torch.cuda.amp.autocast():
        data = data.map(add_predictions, batched=True, batch_size=8)
    print(compute_metrics(data["pred"], data["true"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="microsoft/codebert-base")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoint-185000/pytorch_model.bin")

    args = parser.parse_args()

    main(args)