# 传参
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='database/stage_1/Group_26_2555.test.fasta', help='Import fasta file')
parser.add_argument('-o', '--output', type=str, default='result/', help='Export folders')
parser.add_argument('-b', '--batchSize', type=int, default=128, help='Batch size')
parser.add_argument('-p', '--prob_ARG', type=float, default=0.99, help='Confidence threshold predicted as ARG, Extract fasta')

args = parser.parse_args()

# 输出帮助信息
parser.print_help()

input = args.input
output = args.output
batch_size = args.batchSize
prob_ARG = args.prob_ARG

# 导入包
from Bio import SeqIO
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader

from modelEnsemble.seqGrouping import seqGrouping
from modelEnsemble.getModelStage1 import getModelStage1,AMPDataset,clsid_to_class_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input = 'database/stage_1/stage_1.test.fasta'
# output = 'result/test0219/'
# batch_size = 128
# prob_ARG = 0.99

stage = 'Stage1'

seq_grouped = seqGrouping(input)
print(seq_grouped.keys())

result = []
for group in seq_grouped.keys():
    # group = '0_100'
    print(f'group: {group}')

    data = seq_grouped[group]

    split_string = group.split("_")
    max_seq_len = int(split_string[1])

    dataTest = AMPDataset(data,max_seq_len=max_seq_len)


    if group != '600_1400':
        CNN_model = getModelStage1(model_type='CNN', group=group)
        LSTM_model = getModelStage1(model_type='LSTM', group=group)

        modelList = [CNN_model, LSTM_model]
        keywords = ['CNN_model', 'LSTM_model']
    else:
        CNN_model = getModelStage1(model_type='CNN', group=group)
        modelList = [CNN_model]
        keywords = ['CNN_model']

    for i in range(len(modelList)):
        keyword = keywords[i]
        print(f'{i} {keyword} {group}')

        # 将数据放到 GPU 上
        dataTest, _, _ = torch.utils.data.random_split(
            dataTest, [int(len(dataTest)), 0, 0])
        test_loader = DataLoader(dataTest, batch_size=batch_size, shuffle=False)

        # 初始化空列表来存储值
        ids = []
        probs = []
        labels = []

        model = modelList[i]
        model.to(device)
        with torch.no_grad():
            for batch in test_loader:
                sequence = batch[0].to(device).long()
                id_batch = batch[1]
                pred = model(sequence)
                # 将预测得分转换为概率值
                prob = torch.softmax(pred, dim=1)

                label = pred.argmax(dim=1)

                # 使用当前批次的值扩展列表
                ids.extend(id_batch)
                probs.extend(prob[:, 1].tolist())  # 仅保存prob的第二列
                labels.extend(label.tolist())

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
# 对于相同的 'id' 值，进行合并，选择'prob'中值最小的行
# 按'id'分组，并选择'prob'最小值所在的行
max_prob_rows = extracted_df.groupby('id')['prob'].idxmin()
# 根据索引提取对应的行
finalResult = extracted_df.loc[max_prob_rows]
finalResult = finalResult.sort_values(by='prob', ascending=False)
finalResult = finalResult.reset_index(drop=True)

csvPath = os.path.join(output, f'{stage}.finalResult.csv')
os.makedirs(os.path.dirname(csvPath), exist_ok=True)
finalResult.to_csv(csvPath, index=False)


# 提取出ARG
df_ARG = finalResult[finalResult['class'] == 'ARG']
# 根据阈值筛选
# prob_ARG = 0.99
df_ARG = df_ARG[df_ARG['prob'] > prob_ARG]
csvPath = os.path.join(output, f'{stage}.finalResult.ARG.csv')
df_ARG.to_csv(csvPath, index=False)

# 提取序列并保存
id_list = df_ARG['id'].tolist()
# 用于保存fasta序列的列表
sequences = []
# 遍历fasta文件，将符合条件的序列添加到sequences列表中
for record in SeqIO.parse(input, "fasta"):
    if (record.id in id_list) and len(record.seq) > 100 and len(record.seq) <= 600:
        sequences.append(record)

# 将sequences列表中的序列保存为fasta文件
SeqIO.write(sequences, os.path.join(output, f'{stage}.ARG.fasta'), "fasta")

