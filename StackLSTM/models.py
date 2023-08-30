import pickle

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from stacknn.structs import Stack
from torch import nn
from torch.utils.data import Dataset


class StackLSTM(nn.Module):
    def __init__(self,
                 embedding_size,
                 embedding_dim,
                 hidden_size_controller,
                 hidden_size_stack,
                 batch_size,
                 label_encoder) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.hidden_size_controller = hidden_size_controller
        self.hidden_size_stack = hidden_size_stack
        self.label_encoder = label_encoder

        self.embedding = nn.Embedding(self.embedding_size, embedding_dim, sparse=False)

        self.controller = nn.LSTMCell(input_size=embedding_dim + hidden_size_stack, hidden_size=hidden_size_controller)

        self.output_linear = nn.Linear(in_features=hidden_size_controller, out_features=self.embedding_size)
        self.softmax = nn.Softmax()

        # self.input_buffer = buffers.InputBuffer(batch_size=batch_size, embedding_size=hidden_size_stack)
        # self.output_buffer = buffers.OutputBuffer(batch_size=batch_size, embedding_size=hidden_size_stack)

        self.push_fc = nn.Linear(hidden_size_controller, 1)
        self.pop_fc = nn.Linear(hidden_size_controller, 1)
        self.values_fc = nn.Linear(hidden_size_controller, hidden_size_stack)

        # self.input_fc = nn.Linear(hidden_size_controller, 1)    
        # self.output_fc = nn.Linear(hidden_size_controller, 1)

        self.classifier = nn.Linear(in_features=embedding_size, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        stack = Stack(batch_size=self.batch_size, embedding_size=self.hidden_size_stack)
        embedded_x = self.embedding(x)
        hx, cx, rx = self.init_hidden()
        outputs = []
        for i, curr_x in enumerate(embedded_x):
            cat_x_rx = torch.cat((curr_x, rx), axis=-1)
            hx, cx = self.controller(cat_x_rx, (hx, cx))

            pop = self.pop_fc(hx).sigmoid()
            values = self.values_fc(hx).relu()
            push = self.push_fc(hx).sigmoid()
            rx = stack(values, pop, push)
            outputs.append(self.output_linear(hx).relu())
        return self.classifier(outputs[-1]).sigmoid()

    def init_hidden(self):
        hx = torch.zeros((self.batch_size, self.hidden_size_controller))  # batch, hidden_size
        cx = torch.zeros((self.batch_size, self.hidden_size_controller))
        rx = torch.ones((self.batch_size, self.hidden_size_stack))
        # ox = torch.ones((self.batch_size, self.hidden_size_stack))
        # ix = torch.ones((self.batch_size, self.hidden_size_stack))

        return hx, cx, rx

    def predict(self, outputs):
        encoded_predictions = outputs.argmax(axis=1).numpy()
        translated_result = np.zeros_like(encoded_predictions).astype(object)
        for idx, seq in enumerate(encoded_predictions):
            translated_result[idx] = self.label_encoder.inverse_transform(seq)

        return translated_result


def memoize(func):
    S = {}

    def wrappingfunction(*args):
        if args not in S:
            S[args] = func(*args)
        return S[args]

    return wrappingfunction


class NVDDataset(Dataset):
    def __init__(self, data_path, vocab_path):
        with open(vocab_path, "rb") as input_file:
            vocab = pickle.load(input_file)
        self.le = LabelEncoder()
        # start of code, end of code
        self.le.fit(["<SOC>", "<EOC>"])
        self.le.fit(list(vocab))
        self.vocab_size = len(self.le.classes_)

        with open(data_path, "rb") as input_file:
            self.data = pickle.load(input_file)

    def __len__(self):
        return len(self.data)

    @memoize
    def __getitem__(self, index):
        code = torch.tensor(self.le.transform(self.data[index]["code"]))
        label = torch.tensor(self.data[index]["label"])
        return code, label
