from NarrativeKoGPT2.util.data import sentencePieceTokenizer, toString


def makeDataUnderMaxTokenLen():
  # tokenizer
  sentencepieceTokenizer= sentencePieceTokenizer()

  # Files for read and write
  file = open('/Users/a60058238/Desktop/dev/workspace/nlp/Data/fairy_tale_utf-8.txt', 'r', encoding='utf-8')
  untokenized_file = open('/Users/a60058238/Desktop/dev/workspace/nlp/Data/train_fairy_tale_data_utf8.txt', 'w', encoding='utf-8')

  # Data for saving that will use on training
  untokenized = ""
  tokenized = ""
  data_length = 0
  print("tmp_line: ", sentencepieceTokenizer('\n'))

  # Preprocess datas
  while True:
    line = file.readline()
    if "#####" in line:
      untokenized_file.write(untokenized+'\n')
      untokenized = ""
      data_length = 0
      continue

    if not line:
      untokenized_file.write(untokenized)
      break

    tmp_line = line[:-1]
    tokenized_line = sentencepieceTokenizer(tmp_line)
    tokenized_line_len = len(tokenized_line)
    # print("tmp_line: ",tmp_line)
    # print("tokenized_line_len: ",tokenized_line_len)


    # Data length for writing has to under 1022
    # input data can get 1024 token
    # but we need to use BOS and EOS token
    pre_data_len = data_length
    data_length = data_length + tokenized_line_len # bos와 eos 토큰 갯수 고려 +2

    if data_length >= 1000:
      if pre_data_len != len(sentencepieceTokenizer(untokenized)):
        print('pre_data_len: ', pre_data_len)
        print('len(sentencepieceTokenizer(untokenized)): ', len(sentencepieceTokenizer(untokenized)))
      untokenized_file.write(untokenized+'\n')
      untokenized = ""
      data_length = tokenized_line_len
    if untokenized == "":
      untokenized = tmp_line
    else:
      untokenized = untokenized + " " +tmp_line

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

def checkLineTokenLen(path):
  # tokenizer
  sentencepieceTokenizer= sentencePieceTokenizer()

  # Files for read and write
  untokenized_file = open(path, 'r', encoding='utf-8')

  # Preprocess datas
  while True:
    line = untokenized_file.readline()
    tokenized_line = sentencepieceTokenizer(line)
    # print('line: ', line)
    if len(tokenized_line)>1000:
      print('1024 초과: ',len(tokenized_line))
      print('1024 초과: ',tokenized_line)

    if not line:
      break

def insertBOSEOSToken(fromPath, toPath):
  fromFile = open(fromPath, 'r', encoding='utf-8')
  toFile = open(toPath, 'w', encoding='utf-8')

  while True:
    line = fromFile.readline()
    tmp_line = "<s>" + line[:-1] +"</s>\n"
    toFile.write(tmp_line)

    if not line:
      break
  fromFile.close()
  toFile.close()

if __name__ == "__main__":
    trainDataPath = '/Users/a60058238/Desktop/dev/workspace/nlp/Data/train_fairy_tale_data_utf8.txt'
    tokenedTrainDataPath = '/Users/a60058238/Desktop/dev/workspace/nlp/Data/tokened_train_fairy_tale_data_utf8.txt'
    # execute only if run as a script
    # makeDataUnderMaxTokenLen()
    # insertBOSEOSToken(trainDataPath, tokenedTrainDataPath)
    checkLineTokenLen(tokenedTrainDataPath)
