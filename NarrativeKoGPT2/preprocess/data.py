from NarrativeKoGPT2.util.data import sentencePieceTokenizer, toString

def makeDataUnderMaxTokenLen():
  # tokenizer
  sentencepieceTokenizer= sentencePieceTokenizer()

  # Files for read and write
  file = open('../data/backmyo_novel_1/prerpcessed_bm_novel_utf8_3.txt', 'r', encoding='utf-8')
  untokenized_file = open('../data/backmyo_novel_1/untokenized_bm_data.txt', 'w', encoding='utf-8')
  tokenized_file = open('../data/backmyo_novel_1/tokenized_bm_data.txt', 'w', encoding='utf-8')

  # Data for saving that will use on training
  untokenized = ""
  tokenized = ""
  data_length = 0

  # Preprocess datas
  while True:
    line = file.readline()

    if not line:
      untokenized_file.write(untokenized)
      tokenized_file.write(tokenized)
      break

    tokenized_line = sentencepieceTokenizer(line)

    # Data length for writing has to under 1022
    # input data can get 1024 token
    # but we need to use BOS and EOS token
    if data_length+len(tokenized_line) >= 1022:
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


def getBatchData(batch_size, file_path, tokenizer, vocab):

  batch_data=[]
