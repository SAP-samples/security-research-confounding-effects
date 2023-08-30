import glob
import pickle
from collections import defaultdict

from sctokenizer import CTokenizer
from tqdm import tqdm

train_files_vuln = glob.glob("/Users/i534627/Projects/codeartifactevaluation/train/*_1.c")
train_files_clean = glob.glob("/Users/i534627/Projects/codeartifactevaluation/train/*_0.c")[:len(train_files_vuln)]
test_files_vuln = glob.glob("/Users/i534627/Projects/codeartifactevaluation/test/test/*_1")
test_files_clean = glob.glob("/Users/i534627/Projects/codeartifactevaluation/test/test/*_0")[:len(test_files_vuln)]

tokenizer = CTokenizer()  # this object can be used for multiple source files
vocabulary = defaultdict(int)


def process_vocabulary(files):
    for file in tqdm(files):
        with open(file) as f:
            source = f.read()
            all_tokens = tokenizer.tokenize(source)
            for token in all_tokens:
                vocabulary[token.token_value] = vocabulary[token.token_value] + 1


process_vocabulary(train_files_clean + train_files_vuln + test_files_clean + test_files_vuln)

sorted_dict = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)
vocab = {}
print("before vocab size: ", len(vocabulary))
for k, v in sorted_dict:
    vocab[k] = 1
    if len(vocab) > 10000:
        break

print("vocabulary size: ", len(vocab))


def process_set(files, label):
    dataset = []
    for file in tqdm(files):
        temp_dict = {"name": file}
        with open(file) as f:
            source = f.read()
            all_tokens = tokenizer.tokenize(source)
            tokens = [x.token_value for x in
                      all_tokens if
                      x.token_value in vocab]  # save only a token (token_type and line are dropped). https://pypi.org/project/sctokenizer/
            temp_dict["code"] = tokens
            temp_dict["label"] = label
        dataset.append(temp_dict)
    return dataset


train_set = process_set(train_files_vuln, 1) + process_set(train_files_clean, 0)
test_set = process_set(test_files_vuln, 1) + process_set(test_files_clean, 0)

with open('trainset_preprocessed.pkl', 'wb') as handle:
    pickle.dump(train_set, handle)
with open('testset_preprocessed.pkl', 'wb') as handle:
    pickle.dump(test_set, handle)
with open('vocab.pkl', 'wb') as handle:
    pickle.dump(vocab, handle)
