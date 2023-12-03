import json
import os
import pandas as pd

# 标准格式："<s>Human: "+问题+"\n</s><s>Assistant: "+答案</s>

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ["train", "validation", "test"]
save_path = '/data0/maqi/KGLQA-data/datasets/NCR/ncr_atom_format_raw'
if not os.path.exists(save_path):
    os.makedirs(save_path)
lens = []
for phase in PHASES:
    raw_file = 'dev' if phase == 'validation' else phase
    dataset = json.load(open(f'/data0/maqi/KGLQA-data/datasets/NCR/ncr_raw/{raw_file}_2.json', 'r'))
    out_data = []
    for elem in dataset:
        elem_content = elem["Content"]
        questions = []
        for elem_question in elem["Questions"]:
            options = '\n'.join(elem_question['Choices'])

            instruction = ("请根据下文内容选择出正确的选项,你需要仔细阅读原文,根据原文和你已有的知识选择出正确的选项,你的输出格式为:{'option': '选项', 'answer': '答案'}\n"
                           + f"原文:\n{elem_content}\n问题:\n{elem_question['Question']}\n选项:\n{options}\n"
                           + f"请一步步的思考,根据原文和你已有的知识选择出正确的选项")

            output = {'option': elem_question['Answer'],
                      'answer': elem_question['Choices'][LABEL_TO_ID_DICT[elem_question['Answer']]]}

            fm_data = f"<s>Human: {instruction}\n" + f"</s><s>Assistant: {output}</s>"

            lens.append(len(fm_data))
            out_data.append(fm_data)
    sum(lens) / len(lens)
    # 使用pandas
    df = pd.DataFrame(out_data, columns=['text'])
    df.to_csv(os.path.join(save_path, f"{raw_file}_sft.csv"), index=False)
    print('save to ', os.path.join(save_path, f"{raw_file}_sft.csv"))
