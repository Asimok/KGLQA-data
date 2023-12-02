import sys

sys.path.append('/data0/maqi/KGLQA-data')
from transformers import AutoTokenizer, LlamaTokenizer

from retriever.core.retriever import RocketScorer, Retrieval
import flask


def load_tokenizer_en():
    tokenizer_en_ = AutoTokenizer.from_pretrained(
        '/data0/maqi/huggingface_models/llama-2-7b',
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False
    )
    return tokenizer_en_


def load_tokenizer_zh():
    ckpt_path = '/data0/maqi/N_BELLE/BELLE/train/Version7/st5/'
    tokenizer_zh_ = LlamaTokenizer.from_pretrained(ckpt_path)
    tokenizer_zh_.pad_token_id = 0
    tokenizer_zh_.bos_token_id = 1
    tokenizer_zh_.eos_token_id = 2
    tokenizer_zh_.padding_side = "left"
    return tokenizer_zh_


tokenizer_en = load_tokenizer_en()
tokenizer_zh = load_tokenizer_zh()


def key_sentence_zh():
    scorer = RocketScorer(model_name='zh_dureader_de', batch_size=32)
    Retriever = Retrieval(scorer=scorer, tokenizer=tokenizer_zh)
    return scorer, Retriever


def key_sentence_en():
    scorer = RocketScorer(model_name='v2_nq_de', batch_size=32)
    Retriever = Retrieval(scorer=scorer, tokenizer=tokenizer_en)
    return scorer, Retriever


def get_key_sentence(retrieval, scorer, query, options, context, max_word_count=1536):
    sent_data, word_count = retrieval.get_sent_data(context)

    raw_context, select_idx = retrieval.get_top_sentences_mark(
        query=query,
        sent_data=sent_data,
        opt_data=options,
        max_word_count=max_word_count,
        scorer_=scorer,
    )
    return raw_context, select_idx


if __name__ == '__main__':
    app = flask.Flask(__name__)
    scorer_en, Retriever_en = key_sentence_en()
    scorer_zh, Retriever_zh = key_sentence_zh()


    def test():
        # test
        query = "下列材料相关内容的概括和分析，不正确的一项是"
        options = ['A#幼年的高锟热衷化学实验，后来又迷恋无线电，这段经历表现出的特质对他后来进行光纤通信研究具有重要的作用。',
                   'B#高锟先生为人谦虚，对人和蔼，关心家人，用实际行动支持学生自由发表言论，表现了一位科学家的高尚美德。',
                   'C#文章引用高锟的妻子黄美芸和网友的话，突出了高锟在光纤通信科研领域的重大贡献，表达了对高锟的崇敬之情。', 'D#这篇传记记述了传主高锟人生中的一些典型事件， 通过正面和侧面描写来表现传主，生动形象，真实感人。']
        context = "“光纤之父”高锟一2009 年 10 月 6 日凌晨 3 点，美国硅谷一座公寓里响起电话铃。对方说从瑞典打来，有个教授要与高锟先生通话。 几分钟后， 一年一度的诺贝尔物理学奖即将公布。 高锟仍是睡眼惺忪，“什么？我！啊，很高兴的荣誉呢！ ”说完倒头大睡。发表那篇著名论文《为光波传递设置的介电纤维表面波导管》 -亦即光纤通信诞生之日——十年后， 1976 年，高锟拿到人生中第一个奖项——莫理奖。奖杯是一个水晶碗，以前被拿来装火柴盒，现在则盛满了贝壳，放在书柜上。十多年前的一张行星命名纪念证书，还贴在车库墙上，正下方是换鞋凳。最倒霉的是 1979 年爱立信奖奖牌，料想是被打扫房子的女工顺走了⋯⋯爱立信奖颁奖礼规格与诺贝尔奖相当。1959 年激光发明，令人们开始畅想激光通信的未来，但实际研究困难重重。此时高锟就职于国际电话电报公司设于英国的标准通讯实验室， 他坚信激光通信的巨大潜力， 潜心研究，致力于寻找足够透明的传输介质。妻子黄美芸难以忘怀，那段时间高锟很晚回家，年幼的子女经常要在餐桌前等他吃饭，化哄她：“别生气，我们现在做的是非常振奋人心的事情，有一天它会震惊全世界的。 ”专家们起初认为，材料问题无法逾越。 33 岁的高锟在论文中提出构想， “只要把铁杂质的浓度降至百万分之一， 可以预期制造出在波长 0.6 微米附近损耗为 20dB/km 的玻璃材料”，这一构想一开始并未引起世界关注。 几年间， 面对各种质疑， 高锟不仅游说玻璃制造商制造“纯净玻璃” ，更远行世界各地推广这一构想。 1976 年，第一代 45Mb/s 光纤通信系统建成，如今铺设在地下和海底的玻璃光纤已超过 10 亿公里，足以绕地球 2.5 万圈，并仍在以每小时数千公里的速度增长。二创造力的火花早在生命萌芽期就不时闪现。高锟在上海度过 15 岁前的时光，晚上有私塾老师教他四书五经， 白天则在霞飞路上的顶级贵族学校接受西式教育。 西式学校透出的自由民主科学气息深深影响到了童年时的高锟。 高锟幼年时就对科学充满兴趣， 最热衷化学实验，曾自制灭火筒、焰火、烟花、晒相纸，经手的氰化物号称“足以毒害全城的人” 。危险实验被叫停，他转而又迷上无线电，组装一部有五六个真空管的收音机不在话下。1948 年举家迁往香港， 先是考上预科留英， 工作后辗转英美德诸国， 一步步走向世界。他说：“是孔子的哲学令我成为一名出色的工程师” ， 童蒙时期不明所以背诵的那句“读书将以穷理，将以致用也”，启发他独立思考，也让他受惠终生。1987 年， 他被遴选为香港中文大学第三任校长， 自认使命就是“为师生缔造更大发展空间” 。他觉得，教职员只要有独立思想，就有创造性。面对学生抗议也是如此。一次，高锟正要致辞，有学生爬上台，扬起上书“两天虚假景象，掩饰中大衰相”的长布横额遮盖校徽，扰攘十多分钟后才被保安推下台。典礼后，一位记者问： “校方会不会处分示威的同学？”他平静地说：“处分？我为什么要处分他们？他们有表达意见的自由。 ”三从中大退休后， 63 岁的高锟不甘寂寞，成立高科桥光纤公司，继续科研之路。 《科学时报》记者采访他， 接过的名片上只写着公司主席兼行政总裁的称谓， 全无院士等荣誉称号一一他曾先后当选瑞典皇家工程科学院海外会员、英国工程学会会员、美国国家工程院会员、中国科学院外籍院士。问他何故，他笑笑说， “这就是在搞科技产业化。 ”谦谦蔼蔼，光华内蕴。 “教授就是任谁都可以向他发脾气的那种人”许多接触过高锟的人都这么说。 黄关芸晚年评价高锟是“一个有着最可爱笑容的人” ，她与高锟相识于同一家公司，从此携手 60 载。 1960 年代初正忙于那篇重要论文的他，还经常将换尿布等家务活全包。获得诺奖后， 黄美芸用部分奖金推动阿兹海默公益事业， 次年高锟慈善基金会即告成立。高锟逝世当天，黄关芸在媒体通稿中也特意提到基金会，称之为高锟的“最后遗愿” 。（摘编自《南方人物周刊》 2018 年 9 月 27 日）"
        gold_label = 2
        raw_context, select_idx = get_key_sentence(Retriever_zh, scorer_zh, query, options, context, max_word_count=200)
        print()


    test()


    @app.route("/key_sentence_en", methods=["POST"])
    def key_sentence_en_api():
        params = flask.request.get_json()
        query = params['query']
        options = params['options']
        context = params['context']
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
        raw_context, select_idx = get_key_sentence(Retriever_en, scorer_en, query, deal_options, context, max_word_count)
        # print(raw_context, select_idx)
        return flask.jsonify({"context": raw_context, "select_idx": select_idx})


    @app.route("/key_sentence_zh", methods=["POST"])
    def key_sentence_zh_api():
        params = flask.request.get_json()
        query = params['query']
        options = params['options']
        context = params['context']
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

        raw_context, select_idx = get_key_sentence(Retriever_zh, scorer_zh, query, deal_options, context, max_word_count)
        # print(raw_context, select_idx)
        return flask.jsonify({"context": raw_context, "select_idx": select_idx})


    app.run(host='0.0.0.0', port=27030)
