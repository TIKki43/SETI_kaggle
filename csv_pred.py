import os
import numpy as np
import pandas as pd
import shutil

data = pd.read_csv('train_labels.csv')
# print(data[data.columns[0]].to_list())

seq_idx = []
for i in os.listdir('/0'):
    # seq_idx.clear()
    if i[:-4] in data[data.columns[0]].to_list():
        idx = data[data.columns[0]].to_list().index(i[:-4])
        seq_idx.append(data[data.columns[1]].to_list()[idx])
        if data[data.columns[0]].to_list()[idx] == 0:
            shutil.move(f'train_img (2)/{i}', f'/home/timur/Documents/Projects/SETI/0/{i}')

for i in os.listdir('/train_img (2)'):
    shutil.move(f'train_img (2)/{i}', f'/home/timur/Documents/Projects/SETI/0/{i}')
# print(seq_idx)

zero, one = 1, 0
for i in seq_idx:
    if i == 1:
        one += 1
    else:
        zero += 1
print(one/zero, 'lolo', (one, zero))
