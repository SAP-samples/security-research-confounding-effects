import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def train_one_epoch(training_loader, optimizer, model, loss_fn):
    last_loss = 0.

    for code, label in tqdm(training_loader):
        if len(code) > 1000:
            continue
        print(code.shape)
        optimizer.zero_grad()
        label = label.unsqueeze(1)
        # Make predictions for this batch
        y_hat = model(code)
        loss = loss_fn(y_hat, label)

        loss.backward()
        optimizer.step()

        last_loss += loss.item()

    return last_loss


def decode_predictions(y_hat, model):
    encoded_predictions = y_hat.argmax(axis=1).numpy()
    str_predictions_in_array = encoded_predictions, model
    str_predictions_in_lines = [' '.join(list(x)) for x in str_predictions_in_array]
    return str_predictions_in_lines


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    batch_sze = len(batch)

    padded_vectors = pad_sequence(xx, padding_value=1)
    xx_pad = padded_vectors[:, :batch_sze]
    return xx_pad, torch.stack(yy).float()
