import json
import os
import random

from jsonlines import jsonlines

from utils.io_json import write_jsonl

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ["dev", "train", 'test']
save_path = '/data0/maqi/KGLQA-data/datasets/NCR/random_select/ncr_random_1400_instruct'
if not os.path.exists(save_path):
    os.makedirs(save_path)
lens = []
for phase in PHASES:
    raw_file = 'dev' if phase == 'validation' else phase
    dataset = []
    with open(f'/data0/maqi/KGLQA-data/datasets/NCR/random_select/ncr_random_1400/{phase}.jsonl', 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    out_data = []
    for idx, elem in enumerate(dataset):
        passage, answer, q, options = elem["context"], DICT_TO_LABEL[elem['label']], elem['query'], [
            elem["option_0"], elem["option_1"], elem["option_2"], elem["option_3"]]
        options = [option[4:] for option in options]
        if len(passage) < 10:
            continue
        prefix = ('Read the following passage and questions, then choose the right answer from options, the answer '
                  'should be one of A, B, C, D.\n\n')
        passage = f'<passage>:\n{passage}\n\n'
        question = f'<question>:\n{q}\n\n'
        option = f'<options>:\nA {options[0]}\nB {options[1]}\nC {options[2]}\nD {options[3]}\n\n'
        suffix = f"<answer>:\n"
        prompt = ''.join([prefix, passage, question, option, suffix])

        message = {"conversation_id": idx + 1,
                   "category": "quality",
                   "conversation": [
                       {
                           "human": prompt,
                           "assistant": answer
                       }]
                   }
        out_data.append(message)
    for i in range(5):
        random.shuffle(out_data)
    print(f'save to {os.path.join(save_path, f"{raw_file}.jsonl")}')
    write_jsonl(out_data, os.path.join(save_path, f"{raw_file}.jsonl"))
