import json
import os

from utils.io_json import write_jsonl

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ["train", "dev", "test"]
save_path = '/data0/maqi/KGLQA-data/datasets/NCR/Caption/ncr_normal_caption'
if not os.path.exists(save_path):
    os.makedirs(save_path)
lens = []
for phase in PHASES:
    raw_file = phase
    dataset = []
    with open(f'/data0/maqi/KGLQA-data/datasets/NCR/Caption/ncr_caption/{phase}.jsonl', 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    out_data = []
    for idx, elem in enumerate(dataset):
        options = []
        for i in range(4):
            if elem.get(f"option_{i}", None) is None:
                options.append('None')
            else:
                options.append(elem[f"option_{i}"][2:])

        captions_and_rel = []
        context = []
        for caption_idx, caption in enumerate(elem['caption_data']):
            captions_and_rel.append(' '.join(caption['text']))
            context.append(caption['caption'])

        out_data.append({
            "captions": captions_and_rel,
            "context": context,
            "query": elem['query'],
            "option_0": 'A.' + options[0],
            "option_1": 'B.' + options[1],
            "option_2": 'C.' + options[2],
            "option_3": 'D.' + options[3],
            "label": elem['label']
        })

    write_jsonl(out_data, os.path.join(save_path, f"{raw_file}.jsonl"))
    print('save to: ', os.path.join(save_path, f"{raw_file}.jsonl"))
