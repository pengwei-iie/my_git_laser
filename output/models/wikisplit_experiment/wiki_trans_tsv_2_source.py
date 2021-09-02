import pandas as pd
import jsonlines
import json
import numpy as np
from tqdm import tqdm

file_path = 'pred.tsv'
valid_source_save = 'valid.src'
valid_tgt_save = 'valid.tgt'
train_source_save = 'train.src'
train_tgt_save = 'train.tgt'
test_source_save = 'pred.src'
test_pre_save = 'pred.pre'
test_tgt_save = 'pred.tgt'
# read saved file
pd_sen = pd.read_csv(file_path, sep='\t')
source = []
pre = []
tgt = []
for index, sentences in pd_sen.iterrows():
    source.append(sentences[0])
    pre.append(sentences[1])
    tgt.append(sentences[2])
with open(test_source_save, 'w', encoding='utf-8') as outfile:
    for i in source:
        outfile.write(i)
        outfile.write('\n')
with open(test_pre_save, 'w', encoding='utf-8') as outfile:
    for i in pre:
        outfile.write(i)
        outfile.write('\n')
with open(test_tgt_save, 'w', encoding='utf-8') as outfile:
    for i in tgt:
        outfile.write(i)
        outfile.write('\n')
print(test_tgt_save)
