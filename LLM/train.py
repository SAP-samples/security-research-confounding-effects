import argparse

import torch
import transformers

from datasets import load_from_disk
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, balanced_accuracy_score


# from https://github.com/salesforce/CodeT5/blob/d929a71f98ba58491948889d554f8276c92f98ae/CodeT5/models.py#LL123C1-L181C24
class DefectModel(transformers.PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args, class_weights=None):
        super(DefectModel, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = torch.nn.Linear(config.hidden_size, 2)
        self.args = args
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

    def get_t5_vec(self, source_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.view(-1, self.tokenizer.model_max_length)
        vec = self.get_t5_vec(input_ids, attention_mask=attention_mask)

        logits = self.classifier(vec)
        prob = torch.nn.functional.softmax(logits, dim=-1)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    
    train_set = load_from_disk(args.trainset)
    val_set = load_from_disk(args.valset)

    class_weights = torch.as_tensor([
        len(train_set) / torch.sum(train_set["label"] == 0),
        len(train_set) / torch.sum(train_set["label"] == 1)
    ]).float()
    class_weights = class_weights / torch.sum(class_weights)

    config_kwargs = {"vocab_size": len(tokenizer),
                     "scale_attn_by_inverse_layer_idx": True,
                     "reorder_and_upcast_attn": True,
                     }
    config = transformers.AutoConfig.from_pretrained(args.model, **config_kwargs)
    model = transformers.T5ForConditionalGeneration.from_pretrained(args.model, config=config)
    model = DefectModel(model, config, tokenizer, None, class_weights=class_weights)

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
        num_train_epochs=3,
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
    model.save_pretrained("trained-model")
    tokenizer.save_pretrained("trained-model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="Salesforce/codet5p-220m")
    parser.add_argument("--trainset", type=str, default="train")
    parser.add_argument("--valset", type=str, default="val")
    parser.add_argument("--resume", default=False, action="store_true")

    args = parser.parse_args()

    main(args)