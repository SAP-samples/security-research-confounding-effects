import logging

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
logging.basicConfig(filename='info.log', filemode='w', level=logging.INFO)

train_dataset = NVDDataset(data_path="trainset_preprocessed.pkl", vocab_path="vocab.pkl")
test_dataset = NVDDataset(data_path="testset_preprocessed.pkl", vocab_path="vocab.pkl")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=funcs.pad_collate, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=funcs.pad_collate)

model = StackLSTM(embedding_size=train_dataset.vocab_size,
                  embedding_dim=EMBED_DIM,
                  hidden_size_controller=HIDDEN_SIZE_CONTROLLER,
                  hidden_size_stack=HIDDEN_SIZE_STACK,
                  batch_size=BATCH_SIZE,
                  label_encoder=train_dataset.le)

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=LR)
best_vloss = 0
for epoch in range(EPOCHS):
    logging.info(f'EPOCH {epoch + 1}:')
    model.train(True)
    epoch_loss = funcs.train_one_epoch(training_loader=train_loader,
                                       optimizer=optimizer,
                                       model=model,
                                       loss_fn=loss_fn)
    running_vloss = 0.0
    model.train(False)
    acc = 0
    for i, vdata in tqdm(enumerate(test_loader), total=len(test_loader)):
        code, label = vdata
        if len(code) > 1000:
            continue

        vlogits = model(code)

        label = label.unsqueeze(1)

        running_vloss += loss_fn(vlogits, label).detach().item() / len(test_loader)
        acc += int((vlogits.detach().item() > 0.5) == label.item())
    acc /= len(test_loader)
    logging.info(f'LOSS train {epoch_loss} valid {running_vloss} accuracy {acc}')
    if acc > best_vloss:
        best_vloss = acc
        model_path = f'model_{epoch}'
        torch.save(model.state_dict(), model_path)
