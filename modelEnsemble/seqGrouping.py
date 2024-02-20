#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from Bio import SeqIO

def seqGrouping(input):
    arg_records = [record for record in SeqIO.parse(input, "fasta")
                   if 26 < len(record.seq) <= 1400]
    # 将序列按长度分组
    group = ['0_100', '100_600', '600_1400']
    seq_grouped_records = {length: [] for length in group}

    for record in arg_records:
        length = len(record.seq)
        if 26 < length <= 1400:
            if length <= 100:
                seq_grouped_records['0_100'].append(record)
            elif length <= 600:
                seq_grouped_records['100_600'].append(record)
            else:
                seq_grouped_records['600_1400'].append(record)

    return seq_grouped_records


if __name__ == "__main__":
    import os
    os.chdir('E:/BGI/05.ARG/AR-Compass')

    input = 'database/stage_1/Group_26_2555.test.fasta'
    seq_grouped = seqGrouping(input)
