import argparse

import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_from_disk


# from https://github.com/salesforce/CodeT5/blob/d929a71f98ba58491948889d554f8276c92f98ae/CodeT5/models.py#LL123C1-L181C24
class DefectModel(transformers.PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModel, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer: transformers.PreTrainedTokenizer = tokenizer
        self.classifier = torch.nn.Linear(config.hidden_size, 2)
        self.args = args

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

    model.load_state_dict(torch.load(args.checkpoint))

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
    
    data.save_to_disk("predicted-data")
    predicted = torch.as_tensor(data["pred"]).argmax(dim=-1)
    correct = predicted == torch.as_tensor(data["true"])
    lens = torch.as_tensor([len(content) for content in data["processed_func"]])

    filtered_lens = lens[lens < 10_000].cpu().numpy()
    filtered_correct = correct[lens < 10_000].cpu().numpy()

    fig, ax = plt.subplots()
    counts, edges = np.histogram(filtered_lens, bins=100)
    ax.plot(edges.tolist()[:-1], counts, marker="o")
    ax.set_yscale("linear")
    fig.savefig("lengths.png")

    fig, ax = plt.subplots()
    H, xedges, yedges = np.histogram2d(filtered_lens, filtered_correct, bins=30)
    H_correct = H[:, -1] / np.sum(H, axis=-1)
    ax.plot(xedges[:-1], H_correct, marker="o")
    fig.savefig("accuracy_lengths.png")

    filtered_preds = predicted[lens < 10_000]
    _, xedges = np.histogram(filtered_lens, bins=30)

    filtered_tps = filtered_lens[(filtered_preds == 1) & filtered_correct]
    filtered_tns = filtered_lens[(filtered_preds == 0) & filtered_correct]
    filtered_fps = filtered_lens[(filtered_preds == 1) & ~filtered_correct]
    filtered_fns = filtered_lens[(filtered_preds == 0) & ~filtered_correct]

    tp_binned = np.histogram(filtered_tps, bins=xedges)[0]
    tn_binned = np.histogram(filtered_tns, bins=xedges)[0]
    fp_binned = np.histogram(filtered_fps, bins=xedges)[0]
    fn_binned = np.histogram(filtered_fns, bins=xedges)[0]

    tpr_binned = tp_binned / (tp_binned + fn_binned)
    tnr_binned = tn_binned / (tn_binned + fp_binned)

    bacc_binned = (tpr_binned + tnr_binned) / 2
    fig, ax = plt.subplots()
    ax.plot(xedges[:-1], bacc_binned, marker="o")
    fig.savefig("baccuracy_lengths.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="Salesforce/codet5p-220m",
                        help="TODO")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoint-35000/pytorch_model.bin")
    parser.add_argument("--dataset", type=str, default="test")

    args = parser.parse_args()

    main(args)