import sys

sys.path.append('/data0/maqi/KGLQA-data')

from knowledage_bank.core.config import Caption_Server
from knowledage_bank.core.knowledge_bank import KnowledgeBank
from knowledage_bank.core.captions import Captions
from transformers import LlamaTokenizer

from retriever.core.retriever import RocketScorer
import flask
import math


def load_tokenizer_zh():
    ckpt_path = '/data0/maqi/N_BELLE/BELLE/train/Version7/st5/'
    tokenizer_zh_ = LlamaTokenizer.from_pretrained(ckpt_path)
    tokenizer_zh_.pad_token_id = 0
    tokenizer_zh_.bos_token_id = 1
    tokenizer_zh_.eos_token_id = 2
    tokenizer_zh_.padding_side = "left"
    return tokenizer_zh_


tokenizer_zh = load_tokenizer_zh()


def knowledge_bank_zh():
    scorer_zh = RocketScorer(model_name='zh_dureader_de', batch_size=32)
    Retriever = KnowledgeBank(scorer=scorer_zh, tokenizer=tokenizer_zh)

    return scorer_zh, Retriever


def get_caption(captions_, context, caption_max_seq_length_=128):
    sent_data, word_count = captions_.get_sent_data(context)
    # 分段
    # 分块数
    max_chunk_num = math.ceil(1900 / caption_max_seq_length_)
    # 平均每块大小
    average_chunk_num = math.ceil(word_count / 400)
    chunk_num = min(max_chunk_num, average_chunk_num)
    # 每块大小
    chunk_size = math.ceil(word_count / chunk_num)
    chunks = captions_.get_chunks(sent_data=sent_data, max_chunk_tokens=chunk_size)

    chunk_captions = []
    for idx, chunk in enumerate(chunks):
        # 打印进度
        print(f'process {idx}/{len(chunks)}', end='\r')
        chunk_caption = captions_.get_caption(sent=chunk)
        chunk_captions.append(chunk_caption)
    return chunks, chunk_captions


def get_caption_and_rel(retriever, scorer_, query, options, context_data, captions_sents, max_word_count):
    retriever.scorer = scorer_
    context_data, context_word_count = retriever.get_sent_data(context_data)
    # caption_data, caption_word_count = retriever.get_sent_data(caption_data)
    caption_data = []
    for sent in captions_sents:
        caption_data.append({
            'text': sent,
            'word_count': retriever.get_token_num(sent)
        })

    context_data, contexts_idx, captions_data, captions_idx = retriever.get_top_context_mark(query=query, context_data=context_data, captions_data=caption_data, opt_data=options, max_word_count=max_word_count)
    return context_data, contexts_idx, captions_data, captions_idx


if __name__ == '__main__':
    app = flask.Flask(__name__)
    scorer_zh, Retriever_zh = knowledge_bank_zh()
    captions_zh = Captions(url=f"{Caption_Server}:7036/get_captions", tokenizer=tokenizer_zh, language='zh', max_seq_length=200)


    @app.route("/knowledge_bank_zh", methods=["POST"])
    def knowledge_bank_zh_api():
        params = flask.request.get_json()
        context = params['context']
        caption_max_seq_length = params['caption_max_seq_length']
        chunks, chunk_captions = get_caption(captions_zh, context, caption_max_seq_length)
        return flask.jsonify({"chunks": chunks, "chunk_captions": chunk_captions})


    @app.route("/knowledge_bank_get_rel_zh", methods=["POST"])
    def get_caption_and_rel_zh_api():
        params = flask.request.get_json()
        query = params['query']
        options = params['options']
        context_data = params['context_data']
        caption_data = params['caption_data']
        max_word_count = params['max_word_count']

        # 使用 A. B. C. D. 将options分割
        deal_options = []
        for split_token in ['B#', 'C#', 'D#']:
            temp = str(options).split(split_token)
            if len(temp) > 1:
                deal_options.append(temp[0])
                options = split_token + temp[1]
        if len(temp) > 1:
            deal_options.append(split_token + temp[1])
        context_data, contexts_idx, captions_data, captions_idx = get_caption_and_rel(Retriever_zh, scorer_zh, query, deal_options, context_data=context_data, captions_sents=caption_data, max_word_count=max_word_count)
        return flask.jsonify({"context_data": context_data, "contexts_idx": contexts_idx, "captions_data": captions_data, "captions_idx": captions_idx})


    app.run(host='0.0.0.0', port=27028)
