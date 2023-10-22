import requests


def generate_prompt_single(input_text):
    return "Human: \n" + input_text + "\n\nAssistant:\n"


def make_requests_rel(input_ask):
    url = "http://219.216.64.75:31888/ask"
    data = {
        "input_ask": generate_prompt_single(input_ask),
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 10,
        "num_beams": 1,
        "max_new_tokens": 20,
        "min_new_tokens": 1,
        "repetition_penalty": 1.2
    }
    # 超时时间设为60s
    response = requests.post(url, json=data, timeout=60)
    result_ = response.json()
    return result_['output']


def make_requests_caption(input_ask):
    url = "http://219.216.64.75:31888/ask"
    data = {
        "input_ask": generate_prompt_single(input_ask),
        "temperature": 0.1,
        "top_p": 0.85,
        "top_k": 40,
        "num_beams": 1,
        "max_new_tokens": 500,
        "min_new_tokens": 1,
        "repetition_penalty": 1.2
    }
    # 超时时间设为60s
    response = requests.post(url, json=data, timeout=60)
    result_ = response.json()
    return result_['output']
# print(make_requests("""请你抽取下面句子中的所有实体。记得上世纪50年代中期，我在完全中学读初中时，曾参加年级的一个大会，旨在交流学习经验，师生同登讲台。"""))
