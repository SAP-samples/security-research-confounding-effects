import argparse

import torch
import torch.nn.functional as F
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


# C_SAMPLE = """wefw
# e
# wed
# w
# efwekfwjefiowjfowjdcw
# efwejferihv"""


@torch.no_grad()
def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.T5ForConditionalGeneration.from_pretrained(args.model)

    # data = load_from_disk(args.dataset)
    model.eval()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)

    # tokenized = tokenizer.encode(C_SAMPLE, return_tensors="pt", truncation=True, padding="max_length")
    input_ids = tokenizer([C_SAMPLE] * 8, return_tensors="pt", truncation=True, padding="max_length").input_ids
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    decoder_input_ids = model._shift_right(input_ids)
    decoder_attention_mask = decoder_input_ids.ne(tokenizer.pad_token_id)

    output = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=decoder_attention_mask,
        )
    logits = output.logits
    
    bsz, seq_length = input_ids.size()
    padding = input_ids.eq(tokenizer.pad_token_id)
    neg_log = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), input_ids.view(-1), reduction="none")
    neg_log = neg_log.view(bsz, seq_length)
    neg_log[padding] = 0
    neg_log = torch.sum(neg_log, dim=1) / torch.sum(~padding, dim=1)
    likelihood = torch.exp(-neg_log)
    
    print(likelihood)

    # with torch.cuda.amp.autocast():
    #     data = data.map(add_predictions, batched=True, batch_size=8)
    # print(compute_metrics(data["pred"], data["true"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="Salesforce/codet5p-220m")
    parser.add_argument("--dataset", type=str, default="test")

    args = parser.parse_args()

    main(args)