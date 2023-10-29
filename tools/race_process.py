import json

from datasets import load_dataset
from jsonlines import jsonlines
from tqdm import tqdm

phase = 'middle'
raw_ds = load_dataset("race", phase)
train_ds = raw_ds["train"]
dev_ds = raw_ds["validation"]
test_ds = raw_ds["test"]


# raw_ds.save_to_disk('/Users/maqi/PycharmProjects/KGLQA-data/datasets/race')
def save_to_jsonl(data, dataset_type):
    out_data = []
    for i in tqdm(range(len(data))):
        sample = data[i]
        passage, answer, q, options = sample['article'], sample['answer'], sample['question'], sample['options']
        out_data.append({
            'passage': passage,
            'answer': answer,
            'question': q,
            'options': options
        })
    with open('/Users/maqi/PycharmProjects/KGLQA-data/datasets/race/raw/' + dataset_type + '.jsonl', 'w') as w:
        json.dump(out_data, w)
    print('save to ', '/Users/maqi/PycharmProjects/KGLQA-data/datasets/race/raw/' + dataset_type + '.jsonl')


def trans_dataset(data, dataset_type):
    out_data = []
    for i in tqdm(range(len(data))):
        sample = data[i]
        passage, answer, q, options = sample['article'], sample['answer'], sample['question'], sample['options']
        prefix = ('Read the following passage and questions, then choose the right answer from options, the answer '
                  'should be one of A, B, C, D.\n\n')
        passage = f'<passage>:\n{passage}\n\n'
        question = f'<question>:\n{q}\n\n'
        option = f'<options>:\nA {options[0]}\nB {options[1]}\nC {options[2]}\nD {options[3]}\n\n'
        suffix = f"<answer>:\n"
        prompt = ''.join([prefix, passage, question, option, suffix])
        # print(prompt)

        message = {"conversation_id": i + 1,
                   "category": "race_middle",
                   "conversation": [
                       {
                           "human": prompt,
                           "assistant": answer
                       }]
                   }
        out_data.append(message)
    with jsonlines.open(f"../datasets/race/race_{dataset_type}.jsonl", 'w') as w:
        for line in out_data:
            w.write(line)


save_to_jsonl(train_ds, f'{phase}_train')
save_to_jsonl(test_ds, f'{phase}_test')
save_to_jsonl(dev_ds, f'{phase}_dev')
