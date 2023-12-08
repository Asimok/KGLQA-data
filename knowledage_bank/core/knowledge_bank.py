import random
import re
from collections import OrderedDict, defaultdict
from typing import List

from typing_extensions import override

from retriever.core.retriever import BaseRetrieval, RocketScorer


class KnowledgeBank(BaseRetrieval):
    def __init__(self, scorer=None, tokenizer=None):
        super().__init__(scorer, tokenizer)

    def split_text_into_sentences(self, text):
        # 使用正则表达式将文本按照句子分隔进行拆分
        sentences_ = re.split(r'[。！？；;\n]', text)
        # 使用两个连续空格分割
        sentences_ = [re.split(r'  ', sent_) for sent_ in sentences_]
        # 展开sentences_
        sentences_ = [sent_ for sent_list in sentences_ for sent_ in sent_list]
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

    def get_sent_data(self, raw_context):
        sent_data = []
        word_count = 0
        if isinstance(raw_context, str):
            raw_context = self.split_text_into_sentences(raw_context)
        for idx, sent_obj in enumerate(raw_context):
            token_num = self.get_token_num(sent_obj)
            if token_num == 0:
                continue
            sent_data.append({
                "text": str(sent_obj).strip(),
                "word_count": token_num,
                "idx": idx
            })
            word_count += token_num
        return sent_data, word_count

    def get_sim(self, query, opt_data, context_sents):
        op_scores_idx = []
        for opt_idx, opt in enumerate(opt_data):
            op_scores_idx.extend([(sent_idx, score) for sent_idx, score in enumerate(self.scorer.score(opt, context_sents))])
        qp_scores_idx = [(sent_idx, score) for sent_idx, score in enumerate(self.scorer.score(query, context_sents))]
        return qp_scores_idx, op_scores_idx

    @override
    def get_top_sentences(self, query: str, sent_data: list[dict], opt_data: list):
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
            score = self.scorer.score(opt, [pair[1] for pair in temp_cal_pair])
            for idx, pair in enumerate(temp_cal_pair):
                op_scores_idx.append((pair[0], score[idx]))
        # 计算question 与 sentence 的score
        qp_scores_idx = [(sent_idx, score) for sent_idx, score in enumerate(self.scorer.score(query, sentences))]

        sorted_scores = OrderedDict()
        for idx, score_ in op_scores_idx:
            sorted_scores[idx] = score_
        for idx, score_ in qp_scores_idx:
            sorted_scores[(idx,)] = score_
        sorted_scores = sorted(sorted_scores.items(), key=lambda x: x[1], reverse=True)
        # 维护每个句子的最大得分
        max_score_dict = defaultdict(float)
        for idx, score_ in sorted_scores:
            if isinstance(idx, tuple):
                for sent_idx in idx:
                    max_score_dict[f't_{sent_idx}'] = max(max_score_dict[f't_{sent_idx}'], score_)
            else:
                max_score_dict[f't_{idx}'] = max(max_score_dict[idx], score_)

        return max_score_dict

    def get_top_context(self, query: str, context_data: list[dict], captions_data: list[dict], opt_data: list, max_word_count: int, scorer_: RocketScorer):
        context_sents = [sent['text'] for sent in context_data]
        captions_sents = [sent['text'] for sent in captions_data]

        # 合并context_sents
        merge_context_sents = ''.join(context_sents)
        context_data, word_count = self.get_sent_data(self.split_text_into_sentences(merge_context_sents))
        context_score = self.get_top_sentences(query=query, sent_data=context_data, opt_data=opt_data)

        # 计算question,计算option 与 sentence 的score
        qp_scores_idx_captions, op_scores_idx_captions = self.get_sim(query=query, opt_data=opt_data, context_sents=captions_sents)

        sorted_scores_context = context_score
        sorted_scores_captions = defaultdict(float)

        for idx, score_ in qp_scores_idx_captions:
            sorted_scores_captions[f'c_{idx}'] = max(sorted_scores_captions.get(f'c_{idx}', 0), score_)
        for idx, score_ in op_scores_idx_captions:
            sorted_scores_captions[f'c_{idx}'] = max(sorted_scores_captions.get(f'c_{idx}', 0), score_)

        # 合并sorted_scores_context sorted_scores_captions
        for key, value in sorted_scores_captions.items():
            sorted_scores_context[key] = max(sorted_scores_context.get(key, 0), value)
        # 按value排序
        sorted_scores_merge = sorted(sorted_scores_context.items(), key=lambda x: x[1], reverse=True)
        select_sorted_scores_merge = [sent_idx for sent_idx, score_ in sorted_scores_merge]

        # 组织上下文的逻辑
        # 排序从高到低
        total_word_count = 0
        chosen_sent_indices = set()
        for sent_idx in select_sorted_scores_merge:
            if sent_idx in chosen_sent_indices:
                continue
            if sent_idx.startswith('c_'):
                sent_word_count = captions_data[int(sent_idx[2:])]["word_count"]
            else:
                sent_word_count = context_data[int(sent_idx[2:])]["word_count"]
            if sent_idx not in chosen_sent_indices:
                total_word_count += sent_word_count
            if total_word_count > max_word_count:
                break
            if total_word_count > max_word_count:
                break
            chosen_sent_indices.add(sent_idx)

        chosen_sent_indices = list(OrderedDict.fromkeys(chosen_sent_indices))
        # 排序
        chosen_sent_indices.sort(key=lambda x: int(x[2:]))
        captions = ''
        contexts = ''
        for sent_idx in chosen_sent_indices:
            if sent_idx.startswith('c_'):
                captions += captions_data[int(sent_idx[2:])]["text"]
            else:
                contexts += context_data[int(sent_idx[2:])]["text"]
        return contexts, captions

    def get_top_context_random(self, query: str, context_data: list[dict], captions_data: list[dict], opt_data: list, max_word_count: int, scorer_: RocketScorer):
        context_sents = [sent['text'] for sent in context_data]
        # 合并context_sents
        merge_context_sents = ''.join(context_sents)
        context_data, word_count = self.get_sent_data(self.split_text_into_sentences(merge_context_sents))

        # 随机生成排序
        temp_context_data = []
        for idx in range(len(context_data)):
            temp_context_data.append(f't_{idx}')
        temp_captions_data = []
        for idx in range(len(captions_data)):
            temp_captions_data.append(f'c_{idx}')
        select_sorted_scores_merge = temp_context_data + temp_captions_data
        random.shuffle(select_sorted_scores_merge)

        # 组织上下文的逻辑
        # 排序从高到低
        total_word_count = 0
        chosen_sent_indices = set()
        for sent_idx in select_sorted_scores_merge:
            if sent_idx in chosen_sent_indices:
                continue
            if sent_idx.startswith('c_'):
                sent_word_count = captions_data[int(sent_idx[2:])]["word_count"]
            else:
                sent_word_count = context_data[int(sent_idx[2:])]["word_count"]
            if sent_idx not in chosen_sent_indices:
                total_word_count += sent_word_count
            if total_word_count > max_word_count:
                break
            if total_word_count > max_word_count:
                break
            chosen_sent_indices.add(sent_idx)

        chosen_sent_indices = list(OrderedDict.fromkeys(chosen_sent_indices))
        # 排序
        chosen_sent_indices.sort(key=lambda x: int(x[2:]))
        captions = ''
        contexts = ''
        for sent_idx in chosen_sent_indices:
            if sent_idx.startswith('c_'):
                captions += captions_data[int(sent_idx[2:])]["text"]
            else:
                contexts += context_data[int(sent_idx[2:])]["text"]
        return contexts, captions

    def get_top_context_mark(self, query: str, context_data: list[dict], captions_data: list[dict], opt_data: list, max_word_count: int):
        """
        返回原文 并标识已选句子
        """
        # context_sents = [sent['text'] for sent in context_data]
        captions_sents = [sent['text'] for sent in captions_data]

        context_score = self.get_top_sentences(query=query, sent_data=context_data, opt_data=opt_data)

        # 计算question,计算option 与 sentence 的score
        qp_scores_idx_captions, op_scores_idx_captions = self.get_sim(query=query, opt_data=opt_data, context_sents=captions_sents)

        sorted_scores_context = context_score
        sorted_scores_captions = defaultdict(float)

        for idx, score_ in qp_scores_idx_captions:
            sorted_scores_captions[f'c_{idx}'] = max(sorted_scores_captions.get(f'c_{idx}', 0), score_)
        for idx, score_ in op_scores_idx_captions:
            sorted_scores_captions[f'c_{idx}'] = max(sorted_scores_captions.get(f'c_{idx}', 0), score_)

        # 合并sorted_scores_context sorted_scores_captions
        for key, value in sorted_scores_captions.items():
            sorted_scores_context[key] = max(sorted_scores_context.get(key, 0), value)
        # 按value排序
        sorted_scores_merge = sorted(sorted_scores_context.items(), key=lambda x: x[1], reverse=True)
        select_sorted_scores_merge = [sent_idx for sent_idx, score_ in sorted_scores_merge]

        # 组织上下文的逻辑
        # 排序从高到低

        total_word_count = 0
        chosen_sent_indices = set()
        for sent_idx in select_sorted_scores_merge:
            if sent_idx in chosen_sent_indices:
                continue
            if sent_idx.startswith('c_'):
                sent_word_count = captions_data[int(sent_idx[2:])]["word_count"]
            else:
                sent_word_count = context_data[int(sent_idx[2:])]["word_count"]
            if sent_idx not in chosen_sent_indices:
                total_word_count += sent_word_count
            if total_word_count > max_word_count:
                break
            if total_word_count > max_word_count:
                break
            chosen_sent_indices.add(sent_idx)

        chosen_sent_indices = list(OrderedDict.fromkeys(chosen_sent_indices))
        # 排序
        chosen_sent_indices.sort(key=lambda x: int(x[2:]))
        captions_idx = []
        contexts_idx = []
        for sent_idx in chosen_sent_indices:
            if sent_idx.startswith('c_'):
                captions_idx.append(int(sent_idx[2:]))
            else:
                contexts_idx.append(int(sent_idx[2:]))
        return [sent['text'] for sent in context_data], contexts_idx, [sent['text'] for sent in captions_data], captions_idx
