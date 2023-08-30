import argparse

import torch
import torch.nn.functional as F
import transformers

from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, balanced_accuracy_score


DATASETS = [
    "test",
    "perturbed-data/apply_codestyle_Chromium",
    "perturbed-data/apply_codestyle_Google",
    "perturbed-data/apply_codestyle_LLVM",
    "perturbed-data/apply_codestyle_Mozilla",
    "perturbed-data/apply_cobfuscate",
    "perturbed-data/double_obfuscate",
    "perturbed-data/obfuscate_then_style",
    "perturbed-data/py_obfuscate_then_style",
    "perturbed-data/apply_py_obfuscator",
]


# from https://github.com/salesforce/CodeT5/blob/d929a71f98ba58491948889d554f8276c92f98ae/CodeT5/models.py#LL123C1-L181C24
class DefectModel(transformers.PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args, class_weights=None):
        super(DefectModel, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer: transformers.PreTrainedTokenizer = tokenizer
        self.classifier = torch.nn.Linear(config.hidden_size, 2)
        self.args = args
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

        self.line_seperator_ids = self.tokenizer.convert_tokens_to_ids(
            [token for token in self.tokenizer.vocab if "Ċ" in token]
        )

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

    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None):
        input_ids = input_ids.view(-1, self.tokenizer.model_max_length)
        vec = self.get_t5_vec(input_ids, attention_mask=attention_mask)

        logits = self.classifier(vec)
        prob = torch.nn.functional.softmax(logits, dim=-1)

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return loss, prob
        else:
            return prob



@torch.no_grad()
def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    config_kwargs = {"vocab_size": len(tokenizer),
                     "scale_attn_by_inverse_layer_idx": True,
                     "reorder_and_upcast_attn": True,
                     }
    config = transformers.AutoConfig.from_pretrained(args.model, **config_kwargs)
    model = transformers.T5ForConditionalGeneration.from_pretrained(args.model, config=config)
    model = DefectModel(model, config, tokenizer, None)
    likelihood_model = transformers.T5ForConditionalGeneration.from_pretrained(args.model)

    def compute_metrics(pred, true, likelihood):
        predicted = torch.as_tensor(pred).argmax(-1)
        return {
            "MCC": matthews_corrcoef(true, predicted),
            "F1": f1_score(true, predicted, average='macro'),
            "Acc": accuracy_score(true, predicted),
            "BAcc": balanced_accuracy_score(true, predicted),
            "avg-Likelihood": torch.mean(likelihood),
            "std-likelihood": torch.std(likelihood),
        }

    model.load_state_dict(torch.load(args.checkpoint), strict=False)
    model.eval()
    likelihood_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    likelihood_model.to(device)

    def add_likelihood(sample):
        input_ids = sample["input_ids"].to(device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        decoder_input_ids = likelihood_model._shift_right(input_ids)
        decoder_attention_mask = decoder_input_ids.ne(tokenizer.pad_token_id)
        
        logits = likelihood_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        ).logits
        
        bsz, seq_length = input_ids.size()
        padding = input_ids.eq(tokenizer.pad_token_id)
        neg_log = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), input_ids.view(-1), reduction="none")
        neg_log = neg_log.view(bsz, seq_length)
        neg_log[padding] = 0
        neg_log = torch.sum(neg_log, dim=1) / torch.sum(~padding, dim=1)
        likelihood = torch.exp(-neg_log)
        
        return {"likelihood": likelihood}

    def add_predictions(sample):
        pred = model(sample["input_ids"].to(device)).cpu().tolist()
        true = sample["label"]
        return {"pred": pred, "true": true}
    
    for dataset_path in DATASETS:
        data = load_from_disk(dataset_path)
        prev_columns = data.column_names
        prev_columns.remove("index")
        
        # data = data.sort(["label", "index"], reverse=True).select(range(2000))
        with torch.cuda.amp.autocast():
            data = data.map(add_predictions, batched=True, batch_size=8)
        data = data.map(add_likelihood, batched=True, batch_size=2, remove_columns=prev_columns)
        data.save_to_disk(f"report/likelihood/{dataset_path}")
        print(dataset_path, compute_metrics(data["pred"], data["true"], data["likelihood"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="Salesforce/codet5p-220m",
                        help="TODO")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoint-35000/pytorch_model.bin")
    parser.add_argument("--dataset", type=str, default="test")

    args = parser.parse_args()

    main(args)