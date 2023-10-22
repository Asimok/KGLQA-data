import json
import os

from utils.io_json import write_jsonl, write_json

PHASES = ["train", "validation", "test"]
pass_q = 0
choice_num = {}
LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
out_data = []
lens = []
save_path = '/data0/maqi/quality/datasets/ncr_format'
phase = 'test'
dataset = json.load(open('/data0/maqi/lrqa/datasets/NCR/test_2.json', 'r'))
for elem in dataset:
    elem_content = elem["Content"]
    questions = []
    for elem_question in elem["Questions"]:
        # assert len(elem_question["Choices"]) == 4
        if len(elem_question["Choices"]) != 4:
            pass_q += 1
            choice_num[len(elem_question["Choices"])] = choice_num.get(len(elem_question["Choices"]), 0) + 1
            # # 补全
            for choice in range(len(elem_question["Choices"]), 4):
                elem_question["Choices"].append(DICT_TO_LABEL[choice] + '. 空')
            continue
        questions.append({
            "question": elem_question["Question"],
            "options": elem_question["Choices"],
            "gold_label": LABEL_TO_ID_DICT[elem_question["Answer"]]
        })
        lens.append(len(elem_question["Question"]))
    out_data.append({
        "article": elem_content,
        "questions": questions
    })

write_jsonl(out_data, os.path.join(save_path, f"{phase}.jsonl"))
write_json({"num_choices": 4}, os.path.join(save_path, "config.json"))
