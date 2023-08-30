import re
import argparse

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import transformers
from transformers import RobertaForSequenceClassification


C_SAMPLE = """PHP_FUNCTION(imageconvolution)
{
zval *SIM, *hash_matrix;
zval **var = NULL, **var2 = NULL;
gdImagePtr im_src = NULL;
double div, offset;
int nelem, i, j, res;
float matrix[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}};

if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "radd", &SIM, &hash_matrix, &div, &offset) == FAILURE) {
RETURN_FALSE;
}

ZEND_FETCH_RESOURCE(im_src, gdImagePtr, &SIM, -1, "Image", le_gd);

nelem = zend_hash_num_elements(Z_ARRVAL_P(hash_matrix));
if (nelem != 3) {
php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");
RETURN_FALSE;
}

for (i=0; i<3; i++) {
if (zend_hash_index_find(Z_ARRVAL_P(hash_matrix), (i), (void **) &var) == SUCCESS && Z_TYPE_PP(var) == IS_ARRAY) {
if (Z_TYPE_PP(var) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_PP(var)) != 3 ) {
php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have 3x3 array");
RETURN_FALSE;
}

for (j=0; j<3; j++) {
if (zend_hash_index_find(Z_ARRVAL_PP(var), (j), (void **) &var2) == SUCCESS) {
					SEPARATE_ZVAL(var2);
					convert_to_double(*var2);
					matrix[i][j] = (float)Z_DVAL_PP(var2);
} else {
php_error_docref(NULL TSRMLS_CC, E_WARNING, "You must have a 3x3 matrix");
RETURN_FALSE;
}
}
}
}
res = gdImageConvolution(im_src, matrix, (float)div, (float)offset);

if (res) {
RETURN_TRUE;
} else {
RETURN_FALSE;
}
}
"""

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


def sort_lines(t):
    _, indices = torch.sort(torch.as_tensor(t), descending=True)
    return indices.tolist()

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

DATASETS = [
    "test",
    "perturbed-data/apply_codestyle_Chromium",
    "perturbed-data/apply_codestyle_Google",
    "perturbed-data/apply_codestyle_LLVM",
    "perturbed-data/apply_codestyle_Mozilla",
    "perturbed-data/apply_codestyle_GNU",
]

@torch.no_grad()
def main(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    line_seperator_ids = tokenizer.convert_tokens_to_ids(
        [token for token in tokenizer.vocab if "Ċ" in token]
    )
    
    config = transformers.RobertaConfig.from_pretrained(args.model)
    config.num_labels = 1
    model = transformers.RobertaForSequenceClassification.from_pretrained(args.model, config=config, ignore_mismatched_sizes=True)
    model = Model(model, config, tokenizer, args)

    model.load_state_dict(torch.load(args.checkpoint), strict=False)
    model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
        
    input_ids = tokenizer.encode(C_SAMPLE, padding="max_length", truncation=True, return_tensors="pt").to(device)
    pred, att = model(input_ids, output_attentions=True)

    print("\n".join(map(str, sorted(zip(enumerate(tokenizer.convert_ids_to_tokens(input_ids[0])), att[0].tolist()), key=lambda t: t[1]))))

    print(sort_lines(get_scores_per_line(input_ids, att, line_seperator_ids)[0]))
    print(pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="microsoft/codebert-base")
    parser.add_argument("--checkpoint", type=str, default="model/12heads_linevul_model.bin")

    args = parser.parse_args()

    main(args)