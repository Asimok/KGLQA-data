import json

phases = ['train', 'dev', 'test']
cnt = 0
dataset_name = None
for phase in phases:
    dataset_path = f"/data0/maqi/KGLQA-data/datasets/NCR/ncr_rocketqa_1400/{phase}.jsonl"
    train_data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            train_data.append(json.loads(line))
            if str(train_data[-1]['query']).__contains__('正确') or str(train_data[-1]['query']).__contains__('错误') or str(train_data[-1]['query']).__contains__('不正确'):
                cnt += 1
    ans = cnt / len(train_data)
    print()
