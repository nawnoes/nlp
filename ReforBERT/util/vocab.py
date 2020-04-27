from ReforBERT.util.common import download
from ReforBERT.util.tokenizer import tokenizer
import gluonnlp as nlp


# download vocab
def koBertVocab():
  cachedir='~/reforBert/'

  vocab_info = tokenizer
  vocab_file = download(vocab_info['url'],
                         vocab_info['fname'],
                         vocab_info['chksum'],
                         cachedir=cachedir)
  vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                       padding_token='[PAD]')
  return vocab_b_obj
