import json
import os

from utils.io_json import write_jsonl, write_json

PHASES = ["train", "dev", "test"]
LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}

save_path = '/data0/maqi/KGLQA-data/datasets/NCR/ncr_firefly_format_full'


def process(phase):
    out_data = []
    dataset = json.load(open(f'/data0/maqi/KGLQA-data/datasets/NCR/ncr_raw/{phase}.json', 'r'))
    idx = 0
    for elem in dataset:
        elem_content = elem["Content"]
        for elem_question in elem["Questions"]:
            idx += 1
            if len(elem_question["Choices"]) != 4:
                # # 补全
                for choice in range(len(elem_question["Choices"]), 4):
                    elem_question["Choices"].append(DICT_TO_LABEL[choice] + '. 空')
                continue

            passage, answer, q, options = elem_content, elem_question["Answer"], elem_question, [
                elem_question["Choices"][0], elem_question["Choices"][1], elem_question["Choices"][2],
                elem_question["Choices"][3]]
            options = [option[2:] for option in options]
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

    write_jsonl(out_data, os.path.join(save_path, f"{phase}.jsonl"))
    print('len out_data', len(out_data))
    print(f'save to {os.path.join(save_path, f"{phase}.jsonl")}')


if __name__ == '__main__':
    for phase in PHASES:
        process(phase)
