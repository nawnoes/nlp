from NarrativeKoGPT2.kogpt2.utils import download, tokenizer, get_tokenizer
from gluonnlp.data import SentencepieceTokenizer
import gluonnlp


def sentencePieceTokenizer():
  tok_path = get_tokenizer()
  sentencepieceTokenizer = SentencepieceTokenizer(tok_path)

  return sentencepieceTokenizer


def koGPT2Vocab():
  cachedir = '~/kogpt2/'

  # download vocab
  vocab_info = tokenizer
  vocab_path = download(vocab_info['url'],
                        vocab_info['fname'],
                        vocab_info['chksum'],
                        cachedir=cachedir)

  koGPT2_vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                             mask_token=None,
                                                             sep_token=None,
                                                             cls_token=None,
                                                             unknown_token='<unk>',
                                                             padding_token='<pad>',
                                                             bos_token='<s>',
                                                             eos_token='</s>')
  return koGPT2_vocab

def toString(list):
  if not list:
    return ''
  result = ''

  for i in list:
    result = result + i
  return result