import re
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

FLAW_LINES = [
    "cin>>buf;",
    "char buf[MAXSIZE];"
]

FLAW_LINE_SEP = "/~/"

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

    def get_t5_vec(self, source_ids, attention_mask=None, output_attentions=False):
        if attention_mask is None:
            attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask, output_attentions=output_attentions,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        if output_attentions:
            att = {
                "decoder": outputs["decoder_attentions"],
                "cross": outputs["cross_attentions"],
                "encoder": outputs["encoder_attentions"],
            }
            return vec, att
        return vec

    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None, output_attentions=False):
        assert not (output_attentions and labels is not None), "Currently not implemented (easily fixable)"
        input_ids = input_ids.view(-1, self.tokenizer.model_max_length)
        outs = self.get_t5_vec(input_ids, attention_mask=attention_mask, output_attentions=output_attentions)

        if output_attentions:
            vec, att = outs
        else:
            vec = outs

        logits = self.classifier(vec)
        prob = torch.nn.functional.softmax(logits, dim=-1)

        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return loss, prob
        else:
            if output_attentions:
                return prob, self.get_attention_scores(input_ids, att)
            return prob
    
    def get_attention_scores(self, input_ids: torch.Tensor, att):
        att = att["encoder"] # only encoder
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

    def get_scores_per_line(self, input_ids: torch.Tensor, att: torch.Tensor):
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
                if token in self.line_seperator_ids:
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
        line = re.sub("^\s", "", line)
        line = re.sub("\s$", "", line)
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
    config_kwargs = {"vocab_size": len(tokenizer),
                     "scale_attn_by_inverse_layer_idx": True,
                     "reorder_and_upcast_attn": True,
                     }
    config = transformers.AutoConfig.from_pretrained(args.model, **config_kwargs)
    model = transformers.T5ForConditionalGeneration.from_pretrained(args.model, config=config)
    model = DefectModel(model, config, tokenizer, None)

    def compute_metrics(ranks, preds):
        ranks = torch.as_tensor(ranks)
        preds = torch.as_tensor(preds).argmax(dim=-1)
        tp_ranks = ranks[preds == 1]
        ranks = ranks[ranks.isfinite()]
        tp_ranks = tp_ranks[tp_ranks.isfinite()]
        return {
            "Top10-Acc": torch.sum(ranks < 10) / len(ranks),
            "IFA": torch.mean(ranks),
            "Top100-Acc": torch.sum(ranks < 100) / len(ranks),
            "Top5-Acc": torch.sum(ranks < 5) / len(ranks),
            "TP-Top10-Acc": torch.sum(tp_ranks < 10) / len(tp_ranks),
            "TP-IFA": torch.mean(tp_ranks[tp_ranks.isfinite()]),
            "TP-Top100-Acc": torch.sum(tp_ranks < 100) / len(tp_ranks),
            "TP-Top5-Acc": torch.sum(tp_ranks < 5) / len(tp_ranks),
            "Count": len(ranks),
            "TP": len(tp_ranks),
        }

    model.load_state_dict(torch.load(args.checkpoint))

    # prepare reference for flaw lines
    reference = load_from_disk(args.reference)
    reference.set_format(None, columns=["index", "flaw_line"])
    index_to_flaw_line = {}
    for sample in reference:
        index_to_flaw_line[sample["index"]] = sample["flaw_line"]

    data = load_from_disk(args.dataset)
    data = data.filter(lambda sample: sample["label"] == 1)
    data.set_format("torch", columns=["processed_func", "label", "input_ids", "index"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def add_ranks(sample):
        input_ids = sample["input_ids"].to(device)
        pred, att = model(input_ids, output_attentions=True)
        pred = pred.cpu().tolist()
        
        line_scores = model.get_scores_per_line(input_ids, att)
        sorted_lines_batch = sort_lines_batched(line_scores)
        ranks = []
        for sorted_lines, text, index in zip(sorted_lines_batch, sample["processed_func"], sample["index"]):
            index = index.item()
            if index not in index_to_flaw_line:
                print(f"Warning: skipping index {index} not found")
                continue
            flaws = index_to_flaw_line.get(index)
            if flaws is None:
                ranks.append(float("inf"))
                continue
            flaw_line_indices = get_flaw_indices(text.splitlines(), flaws.split(FLAW_LINE_SEP))
            ranks.append(
                float(min_rank_of_indices(sorted_indices=sorted_lines, searched_indices=flaw_line_indices))
            )
        
        return {"rank": ranks, "pred": pred}

    with torch.cuda.amp.autocast():
        data = data.map(add_ranks, batched=True, batch_size=8)
    print(compute_metrics(data["rank"], data["pred"]))
    successful_idxs = torch.as_tensor(data["index"])[torch.as_tensor(data["rank"]).isfinite()]
    with open(f"CodeT5___{args.dataset.replace('/', '__')}_successful.txt", "w") as f:
        f.write("\n".join(map(str, successful_idxs.tolist())))

    """
    tokenized = tokenizer.encode(
        C_SAMPLE,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    # _, attention = model(tokenized, output_attentions=True)
    # # print(list(zip(attention[0].tolist(), tokenizer.convert_ids_to_tokens(tokenized[0]))))
    # print(model.get_scores_per_line(tokenized, attention))
    # print(sorted_lines(model.get_scores_per_line(tokenized, attention)))
    print(get_flaw_indices(C_SAMPLE.splitlines(), FLAW_LINES))
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="Salesforce/codet5p-220m",
                        help="TODO")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoint-60000/pytorch_model.bin")
    parser.add_argument("--dataset", type=str, default="test")
    parser.add_argument("--reference", type=str, default="test")

    args = parser.parse_args()

    main(args)