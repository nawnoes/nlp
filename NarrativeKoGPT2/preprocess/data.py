from NarrativeKoGPT2.util.data import sentencePieceTokenizer, toString


# 토크나이저
sentencepieceTokenizer= sentencePieceTokenizer()

# 파일
file = open('../data/backmyo_novel_1/prerpcessed_bm_novel_utf8_3.txt', 'r', encoding='utf-8')
untokenized_file = open('../data/backmyo_novel_1/untokenized_bm_data.txt', 'w', encoding='utf-8')
tokenized_file = open('../data/backmyo_novel_1/tokenized_bm_data.txt', 'w', encoding='utf-8')

untokenized = ""
tokenized = ""

data_length = 0

while True:
  line = file.readline()
  if not line:
    untokenized_file.write(untokenized)
    tokenized_file.write(tokenized)
    break

  tokenized_line = sentencepieceTokenizer(line)

  if data_length+len(tokenized_line) >= 1024:
    untokenized_file.write(untokenized+'\n')
    tokenized_file.write(tokenized+'\n')

    untokenized = ""
    tokenized = ""
    data_length = 0

  untokenized = untokenized + line[:-1]
  tokenized = tokenized + toString(tokenized_line)

  data_length = data_length+len(tokenized_line)

file.close()
untokenized_file.close()
tokenized_file.close()

