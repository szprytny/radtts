from subtitle_parser import parse_subtitles_file

path = 'C:\outdir/test.srt'
result = []
for t, *_ in parse_subtitles_file(path):
    result.append(t)

with open('srts.txt', 'a', encoding='utf-8') as fw:
    fw.write('\n'.join(result))