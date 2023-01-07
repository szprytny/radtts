from typing import List
import nltk
nltk.download('punkt')

flat_map = lambda f, xs: [y for ys in xs for y in f(ys)]

max_line_length = 160

replacer_rules = [
    ('\'', ''),
    ('”', '"'),
    ('„', '"'),
    ('“', '"'),
    ('``', '\''),
    ('–', '-'),
    ('—', '-'),
    ('...', '…'),
    ('»', '"'),
    ('’', "'"),
    ('«', '"'),
]

def interpunction_replacer(line: str):
    for rule in replacer_rules:
        line = line.replace(rule[0], rule[1])
    return line

# def split_long_line(line: str):
#     if len(line) < max_line_length:
#         return [line]
    
#     tokens = nltk.regexp_tokenize(line, ',|;|-|…|\:', gaps=True)
#     result = []
#     current_part = ''
#     for token in tokens:
#         if (len(current_part) + len(token) > max_line_length):
#             result.append(current_part + ';')
#             current_part = token
#         else:
#             current_part += token

#     print( result + [current_part])
#     return result + [current_part]

def join_long_line_parts(chunks: List[str]):
    result = []
    output_chunk = ''
    for chunk in chunks:
        if (len(output_chunk) + len(chunk) > max_line_length):
            result.append(output_chunk)
            output_chunk = chunk
        else:
            output_chunk += chunk

    return result + [output_chunk]


def split_long_line(line: str):
    if len(line) < max_line_length:
        return [line]
    
    interpuction = list(',;-…:')

    tokens = nltk.word_tokenize(line, 'polish', preserve_line=False)
    result = []
    output_chunk = ''
    for token in tokens:
        if token in interpuction:
            result.append(output_chunk + token)
            output_chunk = ''
        else:
            output_chunk += f' {token}'
    result.append(output_chunk)

    return join_long_line_parts(result)


with open('C:/magnet/uniwersum fani/ln.txt', encoding='utf-8') as f:
    lines = ' '.join(list(map(lambda x: x.replace('\n', ''), f.readlines())))

result = nltk.sent_tokenize(lines, 'polish')
result = list(flat_map(split_long_line, list(map(interpunction_replacer, result))))

with open(f"C:/magnet/ln_refined.txt", 'w', encoding="utf-8") as f:
    f.write('\n'.join(result))