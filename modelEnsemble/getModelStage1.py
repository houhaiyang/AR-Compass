#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score


amino_acid_map = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'E': 6, 'Q': 7,
                  'G': 8, 'H': 9, 'I':10, 'L':11, 'K':12, 'M':13, 'F':14,
                  'P':15, 'S':16, 'T':17, 'W':18, 'Y':19, 'V':20,
                  'X':21, 'U':22, 'B':23, 'Z':23, 'O':21}

clsid_to_class_name = {0: 'non-ARG',
                       1: 'ARG'}

class AMPDataset(Dataset):
    def __init__(self, data, max_seq_len=1024):
        self.sequences = []
        self.id = []

        # 读取fasta文件
        for record in data:
            seq = str(record.seq).upper()
            seq_num = [amino_acid_map[c] for c in seq]
            seq_padded = seq_num + [0] * (max_seq_len - len(seq))
            self.sequences.append(seq_padded)
            self.id.append(record.id)

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):
        sequence = np.array(self.sequences[idx])
        id = self.id[idx]

        return sequence, id



class CNNClassifier(nn.Module):
    def __init__(self, max_seq_len):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=24, embedding_dim=16, padding_idx=0)
        self.encoder = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pooling = nn.MaxPool1d(kernel_size=max_seq_len)
        self.fc = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.transpose(x, 1, 2)
        x = self.encoder(x)
        x = self.pooling(x)
        x = x.view(-1, 32)
        x = self.fc(x)
        return x


class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=24, embedding_dim=16, padding_idx=0)
        self.encoder = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.encoder(x)
        x = x[:, -1, :]  # 取最后一个时间步
        x = self.fc(x)
        return x


def getModelStage1(model_type='CNN', group='100_600'):
    if model_type == 'CNN':
        split_string = group.split("_")
        max_seq_len = int(split_string[1])
        # print(max_seq_len)
        model_name = f'{model_type}_Group_{group}.pt'
        model = CNNClassifier(max_seq_len)
        model.load_state_dict(torch.load(f'models/Stage1/{model_name}'))
        return model

    elif model_type=='LSTM':
        if group != '600_1400':
            model_name = f'{model_type}_Group_{group}.pt'
            model = LSTMClassifier()
            model.load_state_dict(torch.load(f'models/Stage1/{model_name}'))
            return model

        else:
            print("Stage1 LSTM model group all non-existent")

    else:
        print("model_type error !")


if __name__ == "__main__":
    import os
    os.chdir('E:/BGI/05.ARG/AR-Compass')

    model = getModelStage1(model_type='CNN',length_type='all',group='26_2555')



