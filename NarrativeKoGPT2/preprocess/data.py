from NarrativeKoGPT2.util.data import sentencePieceTokenizer, toString


def makeDataUnderMaxTokenLen():
  # tokenizer
  sentencepieceTokenizer= sentencePieceTokenizer()

  # Files for read and write
  file = open('../data/backmyo_novel_1/prerpcessed_bm_novel_utf8_3.txt', 'r', encoding='utf-8')
  untokenized_file = open('../data/backmyo_novel_1/untokenized_bm_data.txt', 'w', encoding='utf-8')

  # Data for saving that will use on training
  untokenized = ""
  tokenized = ""
  data_length = 0

  # Preprocess datas
  while True:
    line = file.readline()

    if not line:
      untokenized_file.write(untokenized)
      break

    tmp_line = "<s>" + line[:-1] + "</s>"
    tokenized_line = sentencepieceTokenizer(tmp_line)
    tokenized_line_len = len(tokenized_line)
    print("tmp_line: ",tmp_line)
    print("tokenized_line_len: ",tokenized_line_len)

    # Data length for writing has to under 1022
    # input data can get 1024 token
    # but we need to use BOS and EOS token
    data_length = data_length + tokenized_line_len # bos와 eos 토큰 갯수 고려 +2

    if data_length >= 1022:
      untokenized_file.write(untokenized+'\n')

      untokenized = ""
      data_length = tokenized_line_len
    untokenized = untokenized + tmp_line

    # data_length =  # bos와 eos 토큰 갯수 고려 +2
  file.close()
  untokenized_file.close()


def getBatchData(batch_size, file_path, tokenizer, vocab):

  file = open(file_path, 'r', encoding='utf-8')
  while True:
    line = file.readline()
    tokenized_line = tokenizer(line[:-1]) # 마지막 개행 문자 제거
    [vocab[vocab.bos_token], ] + vocab[tokenized_line]
    if not line:
      break

def checkLineTokenLen():
  # tokenizer
  sentencepieceTokenizer= sentencePieceTokenizer()

  # Files for read and write
  untokenized_file = open('/Users/a60058238/Desktop/dev/workspace/nlp/NarrativeKoGPT2/data/backmyo_novel_1/untokenized_bm_data.txt', 'r', encoding='utf-8')

  # Preprocess datas
  while True:
    line = untokenized_file.readline()
    tokenized_line = sentencepieceTokenizer(line)
    # print('line: ', line)
    if len(tokenized_line)>=1020:
      print('1024 초과: ',len(tokenized_line))
    if not line:
      break

if __name__ == "__main__":
    # execute only if run as a script
    # makeDataUnderMaxTokenLen()
    checkLineTokenLen()