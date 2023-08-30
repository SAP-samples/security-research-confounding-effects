import logging
from glob import glob

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import NVDDataset, StackLSTM
from utilities import funcs

HIDDEN_SIZE_CONTROLLER = 8
EMBED_DIM = 164
HIDDEN_SIZE_STACK = 8
LR = 0.0001
BATCH_SIZE = 1
EPOCHS = 50
logging.basicConfig(filename='test.log', filemode='w', level=logging.INFO)
loaders = {}


def load_loader(name):
    train_dataset = NVDDataset(data_path=name, vocab_path="vocab.pkl")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=funcs.pad_collate, shuffle=True)
    return train_loader


train = NVDDataset(data_path="trainset_preprocessed.pkl", vocab_path="vocab.pkl")

model = StackLSTM(embedding_size=train.vocab_size,
                  embedding_dim=EMBED_DIM,
                  hidden_size_controller=HIDDEN_SIZE_CONTROLLER,
                  hidden_size_stack=HIDDEN_SIZE_STACK,
                  batch_size=BATCH_SIZE,
                  label_encoder=train.le)
model.load_state_dict(torch.load("model_7"))
model.eval()
for name in glob("*.pkl"):
    print(name)
    if name.endswith("vocab.pkl") or name.endswith("testset_preprocessed.pkl") or name.endswith(
            "trainset.preprocessed.pkl"):
        continue
    l = load_loader(name)
    acc = 0
    for i, vdata in tqdm(enumerate(l), total=len(l)):
        code, label = vdata
        if len(code) > 1000:
            continue

        vlogits = model(code)

        label = label.unsqueeze(1)

        acc += int((vlogits.detach().item() > 0.5) == label.item())
    acc /= len(l)
    logging.info(f'{name} accuracy {acc}')
