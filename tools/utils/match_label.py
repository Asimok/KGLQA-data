from utils.io_json import read_jsonl, write_jsonl

# LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
base_file = '/data0/maqi/KGLQA-data/datasets/CCLUE/Caption/cclue_caption_and_rel/test.jsonl'
org_file = '/data0/maqi/KGLQA-data/datasets/CCLUE/cclue_rocketqa_1400/test.jsonl'
org_data = read_jsonl(org_file)
org_data = org_data
base_data = read_jsonl(base_file)
match = 0
for i in range(len(org_data)):
    if (org_data[i]['query']).replace('\n', ' ') == (base_data[i]['query']).replace('\n', ' '):
        match += 1
    else:
        print(org_data[i]['query'])
        print(base_data[i]['query'])
        print()
    base_data[i]['label'] = org_data[i]['label']
write_jsonl(base_data, base_file)
print(match)
