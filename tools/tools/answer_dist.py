# 统计数据集分布
import collections

from tqdm import tqdm

from utils.io_json import read_jsonl

# input_base_path = '/data0/maqi/KGLQA-data/datasets/QuALITY/QuALITY.v1.0.1/QuALITY.v1.0.1.htmlstripped.test'
# data = read_jsonl(input_base_path)
# out = []
# cnt = 0
# for row in tqdm(data):
#     for question in row['questions']:
#         cnt += 1
# print(cnt)

input_base_path = '/data0/maqi/KGLQA-data/datasets/QuALITY/predictions/pred.txt'
data = []
with open(input_base_path, 'r') as f:
    for line in f:
        data.append(line.strip().split(',')[1])
distribution = collections.Counter(data)
print(distribution)
