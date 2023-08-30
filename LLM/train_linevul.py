import argparse

import torch
import torch.nn as nn
import transformers

from datasets import load_from_disk
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, balanced_accuracy_score
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, RobertaConfig


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


def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    
    train_set = load_from_disk(args.trainset)
    val_set = load_from_disk(args.valset)

    train_set = train_set.map(lambda row: tokenizer(
        text=row["processed_func"],
        padding="max_length",
        truncation=True,
    ), batched=True)
    train_set.set_format("torch", columns=["input_ids", "label"])

    val_set = val_set.map(lambda row: tokenizer(
        text=row["processed_func"],
        padding="max_length",
        truncation=True,
    ), batched=True)
    val_set.set_format("torch", columns=["input_ids", "label"])

    class_weights = torch.as_tensor([
        len(train_set) / torch.sum(train_set["label"] == 0),
        len(train_set) / torch.sum(train_set["label"] == 1)
    ]).float()
    class_weights = class_weights / torch.sum(class_weights)

    config = RobertaConfig.from_pretrained(args.model)
    config.num_labels = 1
    model = transformers.RobertaForSequenceClassification.from_pretrained(args.model, config=config, ignore_mismatched_sizes=True)
    model = Model(model, config, tokenizer, args)
    # model.load_state_dict(torch.load(args.checkpoint), strict=False)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def compute_metrics(pred):
        """
        Calculates a few helpful metrics
        :param pred: list
        """
        true = pred.label_ids
        predicted = pred.predictions.argmax(-1)
        return {
            "MCC": matthews_corrcoef(true, predicted),
            "F1": f1_score(true, predicted, average='macro'),
            "Acc": accuracy_score(true, predicted),
            "BAcc": balanced_accuracy_score(true, predicted),
        }

    training_args = transformers.TrainingArguments(
        output_dir="model",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=1e-5,
        logging_dir='./logs',
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        logging_steps=5000,
        save_steps=5000,
        save_total_limit=2,
        evaluation_strategy="steps",
        log_level="info",
        seed=1337,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.evaluate()
    torch.save(model.state_dict(), "trained-model.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="microsoft/codebert-base")
    parser.add_argument("--checkpoint", type=str, default="model_standard/linevul/12heads_linevul_model.bin")
    parser.add_argument("--trainset", type=str, default="train")
    parser.add_argument("--valset", type=str, default="val")
    parser.add_argument("--resume", default=False, action="store_true")

    args = parser.parse_args()

    main(args)