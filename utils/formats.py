import re


def clean_string(text):
    return re.sub(r"[\n\t\r]", " ", text).strip()