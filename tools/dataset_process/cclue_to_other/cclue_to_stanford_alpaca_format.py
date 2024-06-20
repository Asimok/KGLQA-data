import csv
import json
import os

from utils.io_json import write_jsonl

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}


def to_normal(filepath, save_path):
    out = []
    with open(filepath, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for id_, row in enumerate(csv_reader):
            _, question, _, context, label, choice0, choice1, choice2, choice3 = row
            # if str(choice0).__contains__('苏逢吉'):
            #     print(f'A#{choice0}\nB#{choice1}\nC#{choice2}\nD#{choice3}')

            fm_data = {
                "instruction": "请仔细阅读原文,根据原文回答问题,选择出正确的选项。" +
                               f"原文:\n{context}\n问题:\n{question}\n选项:\nA {choice0}\nB {choice1}\nC {choice2}\nD {choice3}\n",
                "input": '',
                "output": DICT_TO_LABEL[int(label)]
            }
            out.append(
                fm_data
            )

    write_jsonl(out, save_path)
    print('save to ', save_path)


if __name__ == '__main__':
    PHASES = ["train", "dev", "test"]
    for phase in PHASES:
        filepath = f'/data0/maqi/KGLQA-data/datasets/CCLUE/CCLUE_raw/{phase}.csv'
        save_path = f'/data0/maqi/KGLTQA/datasets/CCLUE_processed/cclue_standford/{phase}.jsonl'
        to_normal(filepath, save_path)
