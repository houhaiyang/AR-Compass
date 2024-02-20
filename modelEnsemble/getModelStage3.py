#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences # 3.9 版本不同，引用方式不同
# from keras.utils import pad_sequences # 3.7
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from Bio import SeqIO
from torch.optim.lr_scheduler import ReduceLROnPlateau



# defining the essential constant values
MAX_NB_CHARS = 24
amino_acids = list("ARNDCEQGHILKMFPSTWYVXUBZ")  # 24 种氨基酸字母
class_num = 6
ar_name_to_label = {'class A': 0,
                    'subclass B1': 1,
                    'subclass B2': 2,
                    'subclass B3': 3,
                    'class C': 4,
                    'class D': 5}

clsid_to_class_name = {0: 'class A',
                       1: 'subclass B1',
                       2: 'subclass B2',
                       3: 'subclass B3',
                       4: 'class C',
                       5: 'class D'}

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

embedding_dim=64
kernel_size=20
pool_kernel_size=0
stride=1


drop_con1=0.5
drop_con2=0.5
drop_con3=0.5


att_dropout_val =0.5
d_a=100
r=10



class embedding_CNN_attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, class_size, dropout_value=0.5,
                 MAX_SEQUENCE_LENGTH=793, kernel_size=20, pool_kernel_size=0, stride=1,
                 channel_size=2048,att_dropout_val=0.5,d_a=100,r=100,drop_con1=0.0,drop_con2=0.0,drop_con3=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_value)
        self.drop_con1 = nn.Dropout(drop_con1)
        self.drop_con2 = nn.Dropout(drop_con2)
        self.drop_con3 = nn.Dropout(drop_con3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, channel_size, kernel_size=kernel_size, stride=stride)
        # https://blog.csdn.net/sunny_xsc1994/article/details/82969867
        self.conv2 = nn.Conv1d(channel_size, channel_size, kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv1d(channel_size, channel_size, kernel_size=kernel_size, stride=stride)

        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
        self.pool_kernel_size = pool_kernel_size

        self.fc = nn.Linear(channel_size * r, class_size)
        self.linear_first = nn.Linear(channel_size, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a, r)
        self.linear_second.bias.data.fill_(0)
        self.att_dropout= nn.Dropout(att_dropout_val)
        self.r = r

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        xb = self.drop_con1(F.relu(self.conv1(x)))
        out = xb
        out = out.permute(0, 2, 1)
        x = torch.tanh(self.linear_first(out))
        x = self.att_dropout(x)
        x = self.linear_second(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        sentence_embeddings = attention @ out
        out = self.fc(sentence_embeddings.view(x.size(0), -1))
        return out


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, x, mask):
        mask = mask.unsqueeze(2)
        scores = torch.matmul(x, x.transpose(1, 2))  # 计算得分
        scores = scores.masked_fill(mask == 0, -1e9)  # 掩码填充
        alpha = torch.softmax(scores, dim=-1)  # 注意力权重
        output = torch.matmul(alpha, x)  # 加权求和
        return output


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, class_size, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)  # 添加更多的LSTM层
        self.attention = Attention()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, class_size)  # 添加更多的全连接层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        embedded = self.embedding(x)
        lstm_out1, _ = self.lstm1(embedded)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out3, _ = self.lstm3(lstm_out2)  # 添加更多的LSTM层

        masked_lstm_out3 = lstm_out3 * mask.unsqueeze(2)
        masked_lstm_out3 = self.attention(masked_lstm_out3, mask)

        lstm_avg = masked_lstm_out3.sum(dim=1) / (mask.sum(dim=1).unsqueeze(1) + 1e-9)  # 平均池化

        lstm_avg = self.dropout(lstm_avg)
        fc_out = self.fc1(lstm_avg)
        fc_out = torch.relu(fc_out)
        out = self.fc2(fc_out)
        return out


def create_separate_sequence(fa_file, amino_acids=amino_acids): # 20种氨基酸
    texts = []
    for index, record in enumerate(SeqIO.parse(fa_file, 'fasta')):
        temp_str = ""
        for item in (record.seq):
            if item in amino_acids:
                temp_str = temp_str + " " + item
        texts.append(temp_str)
    return texts


def create_ids_for_sequences(fastaPath):
    sequences_ids = []
    for record in SeqIO.parse(fastaPath, 'fasta'):
        sequences_ids.append(record.description) # record.description 全部信息，record.id前面的信息
    return sequences_ids

@torch.no_grad()
def get_probas(model, valid_dl):
    model.eval()
    scores = []
    F_softmax = torch.nn.Softmax(dim=1)
    for x, y in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x.cuda())
        scores.append(F_softmax(y_hat.cpu()).numpy())

    return np.concatenate(scores)

def create_set(fastaPath, max_seq_len=793):
    query_texts = create_separate_sequence(fastaPath)
    # labels = create_ids_for_sequences(query_texts, data_fasta, ar_name_to_label)
    train_tokenizer = Tokenizer(num_words=MAX_NB_CHARS)
    train_tokenizer.fit_on_texts(query_texts)  # 适配文本
    query_sequences = train_tokenizer.texts_to_sequences(query_texts)
    query_x = pad_sequences(query_sequences, maxlen=max_seq_len, padding='post')  # 序列填充，后面补0，长度2975

    query_ids = create_ids_for_sequences(fastaPath)

    return query_x,query_ids

def ARGDatasetCNN(fastaPath, max_seq_len=793, batch_size=128):
    query_x_test, query_ids = create_set(fastaPath, max_seq_len=max_seq_len)
    test_dataset = TensorDataset(torch.from_numpy(query_x_test))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader, query_ids

def ARGDatasetLSTM(fastaPath, max_seq_len=793, batch_size=128):
    query_x_test, query_ids = create_set(fastaPath, max_seq_len=max_seq_len)
    test_mask = (query_x_test != 0)
    test_dataset = TensorDataset(torch.from_numpy(query_x_test),
                                 torch.from_numpy(test_mask).bool())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader,query_ids


def getModelStage3(model_type='CNN', group='208_793'):
    if model_type == 'CNN':
        dropout_value = 0.5
        channel_size = 512
        split_string = group.split("_")
        max_seq_len = int(split_string[1])
        model_name = f'{model_type}_Group_{group}.pt'
        model = embedding_CNN_attention(MAX_NB_CHARS, embedding_dim, class_num, dropout_value=dropout_value,
                                        MAX_SEQUENCE_LENGTH=max_seq_len, kernel_size=kernel_size,
                                        pool_kernel_size=pool_kernel_size, stride=stride, channel_size=channel_size,
                                        att_dropout_val=att_dropout_val, d_a=d_a, r=r, drop_con1=drop_con1,
                                        drop_con2=drop_con2, drop_con3=drop_con3)
        model = model.cuda()
        model.load_state_dict(torch.load(f'models/Stage3/{model_name}'))
        return model

    elif model_type=='LSTM':
        embedding_size = 64
        hidden_size = 128
        dropout_value = 0.5
        model_name = f'{model_type}_Group_{group}.pt'
        model = LSTMModel(MAX_NB_CHARS, embedding_size, hidden_size, class_num, dropout_value)
        model = model.cuda()
        model.load_state_dict(torch.load(f'models/Stage3/{model_name}'))
        return model

    else:
        print("model_type error !")


if __name__ == "__main__":
    import os
    os.chdir('E:/BGI/05.ARG/AR-Compass')

    model = getModelStage2(model_type='CNN',length_type='all',group='26_793')



