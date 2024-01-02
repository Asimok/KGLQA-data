"""
检测ncr训练集和cclue测试集合是否存在数据泄漏问题
"""
from utils.io_json import read_jsonl

cclue_dev = '/data0/maqi/KGLQA-data/datasets/CCLUE/cclue_normal/dev.jsonl'
cclue_test = '/data0/maqi/KGLQA-data/datasets/CCLUE/cclue_normal/test.jsonl'

ncr_train = '/data0/maqi/KGLQA-data/datasets/NCR/ncr_format/train.jsonl'

cclue_dev_data = read_jsonl(cclue_dev)
cclue_test_data = read_jsonl(cclue_test)

ncr_train_data = read_jsonl(ncr_train)


def check(cclue_data):
    match = 0
    for i in range(len(cclue_data)):
        for j in range(len(ncr_train_data)):
            if cclue_data[i]['context'] == ncr_train_data[j]['article']:
                for question in ncr_train_data[i]['questions']:
                    if question['question'] == cclue_data[i]['query'] and question['gold_label'] == cclue_data[i]['label']:
                        match += 1
    return match


dev = check(cclue_dev_data)
test = check(cclue_test_data)
print()
