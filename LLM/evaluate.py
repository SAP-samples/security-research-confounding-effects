import argparse

import torch
import transformers

from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, balanced_accuracy_score


C_SAMPLE = """#include <iostream>
using namespace std;

#define MAXSIZE    40
void test(void)
{
        char buf[MAXSIZE];
        cin>>buf;
        cout<<buf<<endl;
}

int main(int argc, char **argv)
{
        test();
        return 0;
}"""


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

    def compute_metrics(pred, true):
        """
        Calculates a few helpful metrics
        :param pred: list
        """
        predicted = torch.as_tensor(pred).argmax(-1)
        return {
            "MCC": matthews_corrcoef(true, predicted),
            "F1": f1_score(true, predicted, average='macro'),
            "Acc": accuracy_score(true, predicted),
            "BAcc": balanced_accuracy_score(true, predicted),
        }

    model.load_state_dict(torch.load(args.checkpoint), strict=False)

    data = load_from_disk(args.dataset)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def add_predictions(sample):
        pred = model(sample["input_ids"].to(device)).cpu().tolist()
        true = sample["label"]
        return {"pred": pred, "true": true}

    with torch.cuda.amp.autocast():
        data = data.map(add_predictions, batched=True, batch_size=8)
    print(compute_metrics(data["pred"], data["true"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="Salesforce/codet5p-220m",
                        help="TODO")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoint-35000/pytorch_model.bin")
    parser.add_argument("--dataset", type=str, default="test")

    args = parser.parse_args()

    main(args)