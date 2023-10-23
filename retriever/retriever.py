import re
from collections import OrderedDict

import rocketqa
import torch


class RocketScorer:
    def __init__(self,
                 model_name='zh_dureader_de',
                 batch_size=128,
                 device=None,
                 ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_cuda = True if device.type == 'cuda' else False
        self.dual_encoder = rocketqa.load_model(model=model_name, use_cuda=use_cuda, batch_size=batch_size)

    def score(self, query: str, para_list: list):
        query_list = [query] * len(para_list)
        dot_products = self.dual_encoder.matching(query=query_list, para=para_list)
        scores = [score_ for score_ in dot_products]
        return scores


class Retrieval:
    def __init__(self, scorer=None, tokenizer=None):
        self.scorer = scorer
        self.tokenizer = tokenizer

    def get_token_num(self, text):
        # return self.tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
        token = self.tokenizer.encode(text=text, add_special_tokens=False)
        return len(token)

    @staticmethod
    def split_text_into_sentences(text):
        # 使用正则表达式将文本按照句子分隔进行拆分
        sentences_ = re.split(r'[。！？；;\n]', text)

        # 去除空白句子
        sentences_ = [s.strip() for s in sentences_ if s.strip()]

        # 合并使用省略号（...）结尾的句子
        merged_sentences = []
        for i in range(len(sentences_)):
            if sentences_[i].endswith('…') and i < len(sentences_) - 1:
                merged_sentence = sentences_[i] + sentences_[i + 1]
                merged_sentences.append(merged_sentence)
            else:
                merged_sentences.append(sentences_[i])

        return merged_sentences

    def get_sent_data(self, raw_text):
        sent_data = []
        word_count = 0
        for idx, sent_obj in enumerate(self.split_text_into_sentences(raw_text)):
            token_num = self.get_token_num(sent_obj)
            sent_data.append({
                "text": str(sent_obj).strip(),
                "word_count": token_num,
                "idx": idx
            })
            word_count += token_num
        return sent_data,word_count

    @staticmethod
    def get_top_sentences(query: str, sent_data: list[dict], opt_data: list, max_word_count: int,
                          scorer_: RocketScorer):
        sentences = [sent['text'] for sent in sent_data]
        op_scores_idx = []
        # 计算  sent_idx    sent_idx-1+sent_idx   sent_idx+sent_idx+1   sent_idx-1+sent_idx+sent_idx+1 与 option的score
        for opt in opt_data:
            temp_cal_pair = []
            for sent_idx, sent in enumerate(sentences):
                if sent_idx != len(sentences) - 1:
                    temp_cal_pair.append(((sent_idx, sent_idx + 1), sentences[sent_idx] + sentences[sent_idx + 1]))
                if sent_idx != 0:
                    temp_cal_pair.append(((sent_idx - 1, sent_idx), sentences[sent_idx - 1] + sentences[sent_idx]))
                if sent_idx != 0 and sent_idx != len(sentences) - 1:
                    temp_cal_pair.append(((sent_idx - 1, sent_idx, sent_idx + 1),
                                          sentences[sent_idx - 1] + sentences[sent_idx] + sentences[sent_idx + 1]))
                temp_cal_pair.append(((sent_idx,), sentences[sent_idx]))
            score = scorer_.score(opt, [pair[1] for pair in temp_cal_pair])
            for idx, pair in enumerate(temp_cal_pair):
                op_scores_idx.append((pair[0], score[idx]))
        # 计算question 与 sentence 的score
        qp_scores_idx = [(sent_idx, score) for sent_idx, score in enumerate(scorer_.score(query, sentences))]

        sorted_scores = OrderedDict()
        for idx, score_ in op_scores_idx:
            sorted_scores[idx] = score_
        for idx, score_ in qp_scores_idx:
            sorted_scores[(idx,)] = score_
        sorted_scores = sorted(sorted_scores.items(), key=lambda x: x[1], reverse=True)

        total_word_count = 0
        chosen_sent_indices = set()
        for sent_idxs, score_ in sorted_scores:
            for sent_idx in sent_idxs:
                if sent_idx in chosen_sent_indices:
                    continue
                sent_word_count = sent_data[sent_idx]["word_count"]
                if sent_idx not in chosen_sent_indices:
                    total_word_count += sent_word_count
                if total_word_count > max_word_count:
                    break
                chosen_sent_indices.add(sent_idx)
            if total_word_count > max_word_count:
                break

        chosen_sent_indices = list(OrderedDict.fromkeys(chosen_sent_indices))
        shortened_article = " ".join(sent_data[sent_idx]["text"] for sent_idx in chosen_sent_indices)
        return shortened_article

    @staticmethod
    def get_top_sentences_only_oq(query: str, sent_data: list[dict], opt_data: list, max_word_count: int,
                                  scorer_: RocketScorer):
        sentences = [sent['text'] for sent in sent_data]
        op_scores_idx = []
        # 计算  sent_idx
        for opt in opt_data:
            temp_cal_pair = []
            for sent_idx, sent in enumerate(sentences):
                temp_cal_pair.append(((sent_idx,), sentences[sent_idx]))
            score = scorer_.score(opt, [pair[1] for pair in temp_cal_pair])
            for idx, pair in enumerate(temp_cal_pair):
                op_scores_idx.append((pair[0], score[idx]))
        # 计算question 与 sentence 的score
        qp_scores_idx = [(sent_idx, score) for sent_idx, score in enumerate(scorer_.score(query, sentences))]

        sorted_scores = OrderedDict()
        for idx, score_ in op_scores_idx:
            sorted_scores[idx] = score_
        for idx, score_ in qp_scores_idx:
            sorted_scores[(idx,)] = score_
        sorted_scores = sorted(sorted_scores.items(), key=lambda x: x[1], reverse=True)

        total_word_count = 0
        chosen_sent_indices = set()
        for sent_idxs, score_ in sorted_scores:
            for sent_idx in sent_idxs:
                if sent_idx in chosen_sent_indices:
                    continue
                sent_word_count = sent_data[sent_idx]["word_count"]
                if sent_idx not in chosen_sent_indices:
                    total_word_count += sent_word_count
                if total_word_count > max_word_count:
                    break
                chosen_sent_indices.add(sent_idx)
            if total_word_count > max_word_count:
                break

        chosen_sent_indices = list(OrderedDict.fromkeys(chosen_sent_indices))
        shortened_article = " ".join(sent_data[sent_idx]["text"] for sent_idx in chosen_sent_indices)
        return shortened_article
