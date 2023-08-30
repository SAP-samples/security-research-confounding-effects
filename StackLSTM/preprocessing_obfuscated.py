import glob
import pickle
from collections import defaultdict

from sctokenizer import CTokenizer
from tqdm import tqdm

tokenizer = CTokenizer()  # this object can be used for multiple source files
vocabulary = defaultdict(int)

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
print("vocabulary size: ", len(vocab))


def process_set(files, label):
    dataset = []
    for file in tqdm(files):
        temp_dict = {"name": file}
        with open(file) as f:
            source = f.read()
            all_tokens = tokenizer.tokenize(source)
            tokens = [x.token_value for x in all_tokens if x.token_value in vocab]
            temp_dict["code"] = tokens
            temp_dict["label"] = label
        dataset.append(temp_dict)
    return dataset


for obfus_method in glob.glob("/Users/i534627/Projects/codeartifactevaluation/test/*"):
    if obfus_method.endswith("test"):
        continue
    files_vuln = glob.glob("{}/*_1".format(obfus_method))
    files_clean = glob.glob("{}/*_0".format(obfus_method))[:len(files_vuln)]
    name = obfus_method.split("/")[-1]
    print(name, len(files_vuln), len(files_clean))
    test_set = process_set(files_vuln, 1) + process_set(files_clean, 0)
    with open("{}.pkl".format(name), 'wb') as handle:
        pickle.dump(test_set, handle)

# with open('trainset_preprocessed.pkl', 'wb') as handle:
#    pickle.dump(train_set, handle)
# with open('testset_preprocessed.pkl', 'wb') as handle:
#    pickle.dump(test_set, handle)
