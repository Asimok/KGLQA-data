import json
import os
import pandas as pd

# 标准格式："<s>Human: "+问题+"\n</s><s>Assistant: "+答案</s>

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ["train", "validation", "test"]
save_path = '/data0/maqi/KGLQA-data/datasets/NCR/ncr_atom_format_1536'
if not os.path.exists(save_path):
    os.makedirs(save_path)
lens = []
for phase in PHASES:
    raw_file = 'dev' if phase == 'validation' else phase
    dataset = []
    with open(f'/data0/maqi/KGLQA-data/datasets/NCR/ncr_normal_format_for_train_1536_no_qo/{phase}.jsonl', 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    out_data = []
    for elem in dataset:
        elem_content = elem["context"]

        options = f'{elem["option_0"]}\n{elem["option_1"]}\n{elem["option_2"]}\n{elem["option_3"]}'

        instruction = ("请根据下文内容选择出正确的选项,你需要仔细阅读原文,根据原文和你已有的知识选择出正确的选项,你的输出格式为:{'option': '选项', 'answer': '答案'}\n"
                       + f"原文:\n{elem_content}\n问题:\n{elem['query']}\n选项:\n{options}\n"
                       + f"请一步步的思考,根据原文和你已有的知识选择出正确的选项")

        output = {'option': DICT_TO_LABEL[elem['label']],
                  'answer': elem[f"option_{elem['label']}"]}

        fm_data = f"<s>Human: {instruction}\n" + f"</s><s>Assistant: {output}</s>"

        lens.append(len(fm_data))
        out_data.append(fm_data)
    sum(lens) / len(lens)
    # 使用pandas
    df = pd.DataFrame(out_data, columns=['text'])
    df.to_csv(os.path.join(save_path, f"{raw_file}_sft.csv"), index=False)
    print('save to: ', os.path.join(save_path, f"{raw_file}_sft.csv"))
