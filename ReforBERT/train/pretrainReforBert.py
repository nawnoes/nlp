import os
import json
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
from transformers import BertTokenizer, PreTrainedTokenizer
from fairseq.optim.adafactor import Adafactor

from gluonnlp.data import SentencepieceTokenizer
from ReforBERT.dataloader.wiki import WikiDataset
from ReforBERT.util.tokenizer import koBERTTokenizer
from ReforBERT.util.vocab import koBertVocab
from ReforBERT.reformer import Reformer, ReformerLM



if __name__=='__main__':
  # 데이터 셋
  data_path = '/Users/a60058238/Desktop/dev/wokspace/nlp/Data/kowiki'
  dataset = WikiDataset(path='D:/data/enwiki')

  # 기존 BERT 토크나이저
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

  # KoBERT 토크나이저
  tok_path = koBERTTokenizer()
  sentencepieceTokenizer = SentencepieceTokenizer(tok_path)

  vocab_size = 8002 # vocab 크기
  max_seq_len = 512 # 최대 입력 길이
  embedding_size = 768 # 임베딩 사이

  batch_size = 4 # 학습 시 배치 크기