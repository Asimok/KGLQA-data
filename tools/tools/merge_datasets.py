import os
import random
from jsonlines import jsonlines

dataset_quality = '/data0/maqi/KGLQA-data/datasets/QuALITY/quality_rocketqa_4096_instruct/train.jsonl'
dataset_race_train = '/data0/maqi/KGLQA-data/datasets/RACE/race_train.jsonl'
# dataset_race_dev = '/data0/maqi/KGLQA-data/datasets/race/race_dev.jsonl'

# dataset_quality = '/data0/maqi/KGLQA-data/datasets/NCR/ncr_firefly_format_1536/train.jsonl'
# dataset_race_train = '/data0/maqi/KGLQA-data/datasets/CCLUE_processed/cclue_firefly/train.jsonl'
# dataset_race_dev = '/data0/maqi/KGLQA-data/datasets/CCLUE_processed/cclue_firefly/dev.jsonl'
# dataset_race_test = '/data0/maqi/KGLQA-data/datasets/CCLUE_processed/cclue_firefly/test.jsonl'

# 合并两个数据集
with jsonlines.open(dataset_quality, 'r') as f:
    quality_data = [line for line in f]

with jsonlines.open(dataset_race_train, 'r') as f:
    race_data = [line for line in f]

# with jsonlines.open(dataset_race_dev, 'r') as f:
#     race_data += [line for line in f]
#
# with jsonlines.open(dataset_race_test, 'r') as f:
#     race_data += [line for line in f]

merge_data = quality_data + race_data
# 随机打乱


random.shuffle(merge_data)
random.shuffle(merge_data)
random.shuffle(merge_data)
# 保存
save_path = '/data0/maqi/KGLQA-data/datasets/QuALITY/quality_rocketqa_4096_and_race_instruct/train.jsonl'
# 文件夹不存在 则创建
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
with jsonlines.open(save_path, 'w') as f:
    for line in merge_data:
        f.write(line)
print(f'save to {save_path}')
