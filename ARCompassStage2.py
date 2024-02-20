# 传参
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='result/test0122/Stage1.ARG.fasta', help='Import fasta file')
parser.add_argument('-o', '--output', type=str, default='result/', help='Export folders')
parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('-p', '--prob_beta_lactam', type=float, default=0.9, help='Confidence threshold predicted as beta_lactam, Extract fasta')

args = parser.parse_args()
parser.print_help()

input = args.input
output = args.output
batch_size = args.batch_size
prob_beta_lactam = args.prob_beta_lactam


# 导入包
import numpy as np
import pandas as pd
import os
import torch
from Bio import SeqIO

from modelEnsemble.getModelStage2 import getModelStage2,ARGDatasetCNN,ARGDatasetLSTM,clsid_to_class_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache() # 释放PyTorch使用的GPU显存

# input = 'result/test0219/Stage1.ARG.fasta'
# output = 'result/test0219/'

# input = 'database/stage_2/Group_100_600.test.fasta'
# output = 'result/test0219-stage_2/'

# batch_size = 128
# prob_beta_lactam = 0.9

stage = 'Stage2'

groups = ['100_600']
result = []

for group in groups:
    # group = '100_600'
    print(f'group: {group}')

    # data = seq_grouped[group]
    split_string = group.split("_")
    max_seq_len = int(split_string[1])

    fastaPath = input

    data_CNN, seq_ids = ARGDatasetCNN(fastaPath, max_seq_len=max_seq_len, batch_size=batch_size)
    data_LSTM, _ = ARGDatasetLSTM(fastaPath, max_seq_len=max_seq_len, batch_size=batch_size)

    CNN_model = getModelStage2(model_type='CNN', group=group)
    LSTM_model = getModelStage2(model_type='LSTM', group=group)

    modelList = [CNN_model, LSTM_model]
    keywords = ['CNN_model', 'LSTM_model']

    for i in range(len(modelList)):
        keyword = keywords[i]
        print(f'{i} {keyword} {group}')
        # 模型类型
        model_type = keyword.split("_")[0]

        if i == 0:
            dataTest = data_CNN
        elif i == 1:
            dataTest = data_LSTM
        else:
            print('Data Error !')

        test_loader = dataTest

        # 初始化空列表来存储值
        ids = seq_ids
        probs = []
        labels = []

        model = modelList[i]
        model.to(device)
        model.eval()

        if model_type == 'CNN':
            with torch.no_grad():
                for x in test_loader:
                    x = x[0].long().cuda()
                    # y = y.long().cuda()
                    y_hat = model(x)

                    preds = torch.argmax(y_hat, dim=1)
                    preds = preds.detach().cpu().numpy()
                    labels.extend(preds.tolist())

                    # 将预测得分转换为概率值
                    prob = torch.softmax(y_hat, dim=1)
                    prob = prob.detach().cpu().numpy()
                    prob = np.max(prob, axis=1)
                    probs.extend(prob.tolist())

        elif model_type == 'LSTM':
            with torch.no_grad():
                for x, mask in test_loader:
                    x = x.long().cuda()
                    # y = y.long().cuda()
                    mask = mask.float().cuda()
                    y_hat = model(x, mask)

                    preds = torch.argmax(y_hat, dim=1)
                    preds = preds.detach().cpu().numpy()
                    labels.extend(preds.tolist())

                    # 将预测得分转换为概率值
                    prob = torch.softmax(y_hat, dim=1)
                    prob = prob.detach().cpu().numpy()
                    prob = np.max(prob, axis=1)
                    probs.extend(prob.tolist())


        else:
            print('model_type Error !!!')

        # 创建包含数据的字典
        data = {'id': ids, 'prob': probs, 'label': labels}
        # 从字典创建数据框
        df = pd.DataFrame(data)
        # 根据 label 转化为类型名
        df['label'] = df['label'].astype(int)
        df['class'] = df['label'].map(clsid_to_class_name)
        df['group'] = group
        df['model'] = keyword

        # csvPath = os.path.join(output, f'{stage}.Group_{group}.{keyword}.csv')
        # os.makedirs(os.path.dirname(csvPath), exist_ok=True)
        # df.to_csv(csvPath, index=False)

        result.append(df)

# 将数据框按行合并
merged_df = pd.concat(result, axis=0, ignore_index=True)
csvPath = os.path.join(output, f'{stage}.Group_all.merged.csv')
os.makedirs(os.path.dirname(csvPath), exist_ok=True)
merged_df.to_csv(csvPath, index=False)

# 从 merged_df 中提取列
extracted_df = merged_df[['id', 'prob', 'label', 'class']]
# 对于相同的 'id' 值，进行合并，选择'prob'中值最大的行
# 按'id'分组，并选择'prob'最大值所在的行
max_prob_rows = extracted_df.groupby('id')['prob'].idxmax()
# 根据索引提取对应的行
finalResult = extracted_df.loc[max_prob_rows]
finalResult = finalResult.sort_values(by='prob', ascending=False)
finalResult = finalResult.reset_index(drop=True)

csvPath = os.path.join(output, f'{stage}.finalResult.csv')
os.makedirs(os.path.dirname(csvPath), exist_ok=True)
finalResult.to_csv(csvPath, index=False)


# 提取出 beta_lactam
df_beta_lactam = finalResult[finalResult['class'] == 'beta_lactam']
# 根据阈值筛选
df_beta_lactam = df_beta_lactam[df_beta_lactam['prob'] > prob_beta_lactam]
csvPath = os.path.join(output, f'{stage}.finalResult.beta_lactam.csv')
df_beta_lactam.to_csv(csvPath, index=False)

# 提取序列并保存
id_list = df_beta_lactam['id'].tolist()
# 用于保存fasta序列的列表
sequences = []
# 遍历fasta文件，将符合条件的序列添加到sequences列表中
for record in SeqIO.parse(input, "fasta"):
    if (record.id in id_list) and len(record.seq) > 250 and len(record.seq) <= 400:
        sequences.append(record)

# 将sequences列表中的序列保存为fasta文件
SeqIO.write(sequences, os.path.join(output, f'{stage}.beta_lactam.fasta'), "fasta")
