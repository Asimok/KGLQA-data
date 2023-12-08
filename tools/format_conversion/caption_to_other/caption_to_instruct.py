import json
import os
import random

from jsonlines import jsonlines

from utils.io_json import write_jsonl

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ['dev', 'train', 'test']
LANGUAGES = 'zh'
save_path = '/data0/maqi/KGLQA-data/datasets/CCLUE/Caption/cclue_caption_and_rel_instruct'
if not os.path.exists(save_path):
    os.makedirs(save_path)
lens = []
for phase in PHASES:
    raw_file = 'dev' if phase == 'validation' else phase
    dataset = []
    input_file = f'/data0/maqi/KGLQA-data/datasets/CCLUE/Caption/cclue_caption_and_rel/{phase}.jsonl'
    # input_file = "/data0/maqi/KGLQA-data/datasets/RACE/race_caption_and_rel/all_train.jsonl"
    with open(input_file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    out_data = []
    for idx, elem in enumerate(dataset):
        if elem['label'] is None:
            elem['label'] = 1
        passage_entry, caption_entry, answer, query, options = elem["context"], elem["captions"], DICT_TO_LABEL[elem['label']], elem['query'], [
            elem["option_0"], elem["option_1"], elem["option_2"], elem["LangChain"]]
        options = [option[4:] for option in options]
        # if len(passage_entry) < 10:
        #     continue
        if LANGUAGES == 'en':
            prefix = (
                'Read the following passage, summary and question, then choose the right answer from options, the answer '
                'should be one of A, B, C, D.\n\n')
            passage = f'<passage>:\n{passage_entry}\n\n'
            caption = f'<summary>:\n{caption_entry}\n\n'
            question = f'<question>:\n{query}\n\n'
            option = f'<options>:\nA {options[0]}\nB {options[1]}\nC {options[2]}\nD {options[3]}\n\n'
            suffix = f"<answer>:\n"
        else:
            prefix = (
                '阅读以下段落、摘要和问题，然后从选项中选择正确答案，答案应为A、B、C、D中的一个。\n\n')
            passage = f'<段落>:\n{caption_entry}\n\n'
            caption = f'<摘要>:\n{passage_entry}\n\n'
            question = f'<问题>:\n{query}\n\n'
            option = f'<选项>:\nA {options[0]}\nB {options[1]}\nC {options[2]}\nD {options[3]}\n\n'
            suffix = f"<答案>:\n"
        prompt = ''.join([prefix, passage, caption, question, option, suffix])

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
