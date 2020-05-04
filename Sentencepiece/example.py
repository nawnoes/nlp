import sentencepiece as spm

# Vocab 생
def makeSentencepieceVocab():
  corpus = "kowiki.txt"
  prefix = "kowiki"
  vocab_size = 8000
  spm.SentencePieceTrainer.train(
    f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +
    " --model_type=bpe" +
    " --max_sentence_length=999999" +  # 문장 최대 길이
    " --pad_id=0 --pad_piece=[PAD]" +  # pad (0)
    " --unk_id=1 --unk_piece=[UNK]" +  # unknown (1)
    " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence (2)
    " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence (3)
    " --user_defined_symbols=[SEP],[CLS],[MASK]")  # 사용자 정의 토큰

# 생성된 Vocab 테스트
def testVocab():
  vocab_file = "../Data/kowiki/kowiki.model"
  vocab = spm.SentencePieceProcessor()
  vocab.load(vocab_file)

  lines= [
    "겨울이 되어서 날씨가 무척 추워요.",
    "이번 성탄절은 화이트 크리스마스가 될까요?",
    "겨울에 감기 조심하시고 행복한 연말 되세요."
  ]
  for line in lines:
    pieces = vocab.encode_as_pieces(line)
    ids = vocab.encode_as_ids(line)
    print(line)
    print(pieces)
    print(ids)
    print()

if __name__=='__main__':
  testVocab()