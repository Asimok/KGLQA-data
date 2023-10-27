PREFIX_CH = '你是一个乐意助人的助手，你非常擅长做阅读理解，判断问题和段落的相关性。'
PREFIX_EN = 'You are a helpful assistant who is very skilled at reading comprehension and determining the relevance between question and paragraph.'
CAPTION_PROMPT_ZH = '请用信息量丰富的文字总结一下这段文本:'
CAPTION_PROMPT_EN = 'Please summarize the text with informative words:'


def get_caption_format(language, passage):
    if language == 'zh':
        return f'{CAPTION_PROMPT_ZH}\n{passage}'
    elif language == 'en':
        return f'{CAPTION_PROMPT_EN}\n{passage}'


# def mark_relativity(language, query, passage, mark_type='query'):
#     if mark_type == 'query':
#         if language == 'zh':
#             return (
#                 f'请判断以下段落是否与问题有一定的关联，输出"yes" 或者 "no":\n'
#                 f'段落:\n{passage}'
#                 f'问题:\n{query}\n')
#         elif language == 'en':
#             return (
#                 f'Please determine whether the following paragraph is related to the question. If there is a connection, output "yes" or "no":\n'
#                 f'passage:\n{passage}'
#                 f'question:\n{query}\n')
#     elif mark_type == 'options':
#         if language == 'zh':
#             return (
#                 f'请判断以下段落是否与选项有一定的关联，输出"yes" 或者 "no":\n'
#                 f'段落:\n{passage}'
#                 f'选项:\n{query}\n')
#         elif language == 'en':
#             return (
#                 f'Please determine whether the following paragraph is related to the options provided. output "yes" or "no":\n'
#                 f'passage:\n{passage}'
#                 f'options:\n{query}\n')

def mark_relativity(language, query, passage, mark_type='query'):
    if mark_type == 'query':
        if language == 'zh':
            return (
                f'{PREFIX_CH}'
                f'请判断以下段落是否可以作为解答问题的线索，如果可以请输出"yes" 否则输出 "no":\n'
                f'段落:\n{passage}'
                f'问题:\n{query}\n')
        elif language == 'en':
            return (
                f'{PREFIX_EN}'
                f'Please determine whether the following paragraph can serve as a clue to answer the question. If it can, output "yes"; otherwise, output "no":\n'
                f'paragraph:\n{passage}'
                f'query:\n{query}\n')
    elif mark_type == 'options':
        if language == 'zh':
            return (
                f'{PREFIX_CH}'
                f'请判断以下段落是否与提供的选项有关联，如果是请输出"yes" 否则输出 "no":\n'
                f'段落:\n{passage}'
                f'选项:\n{query}\n')
        elif language == 'en':
            return (
                f'{PREFIX_EN}'
                f'Determine whether the following paragraph is related to the provided options. If it is, output "yes"; otherwise, output "no"\n'
                f'passage:\n{passage}'
                f'options:\n{query}\n')
