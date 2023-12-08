import csv

from utils.io_json import write_jsonl


def to_normal(filepath, save_path):
    out = []
    with open(filepath, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for id_, row in enumerate(csv_reader):
            _, question, _, context, label, choice0, choice1, choice2, choice3 = row
            out.append({
                "context": context,
                "query": question,
                "option_0": choice0,
                "option_1": choice1,
                "option_2": choice2,
                "option_3": choice3,
                "label": int(label),
            })
    write_jsonl(out, save_path)
    print('save to ', save_path)


if __name__ == '__main__':
    PHASES = ["train", "dev", "test"]
    for phase in PHASES:
        filepath = f'../../datasets/CCLUE/{phase}.csv'
        save_path = f'/data0/maqi/KGLTQA/datasets/CCLUE_processed/cclue_normal/{phase}.jsonl'
        to_normal(filepath, save_path)
