import torch
import torch.nn.functional as F
from transformers import BertTokenizer, PreTrainedTokenizer
from ReforBERT.util.tokenizer import koBERTTokenizer
from gluonnlp.data import SentencepieceTokenizer, SentencepieceDetokenizer
from gluonnlp.data import BERTSPTokenizer
from ReforBERT.util.vocab import koBertVocab



def orgin_mask_tokens(tokenizer, inputs: torch.Tensor, mlm_probability=0.15, pad=True):
  """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
  """ 
  Masked Language Model을 위한 마스킹데이터 생성

  마스킹된 입력 input과
  마스킹의 정답인 label을 반환
  """
  # 라벨 생성
  labels = inputs.clone()

  # mlm_probability defaults to 0.15 in Bert
  probability_matrix = torch.full(labels.shape, mlm_probability)

  # sentencepiece 토크나이저에서
  special_tokens_mask = [
    tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
  ]
  print(special_tokens_mask)

  probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
  if tokenizer._pad_token is not None:
    padding_mask = labels.eq(tokenizer.pad_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
  masked_indices = torch.bernoulli(probability_matrix).bool()
  labels[~masked_indices] = -100  # We only compute loss on masked tokens

  # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
  indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
  inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

  # 10% of the time, we replace masked input tokens with random word
  indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
  random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
  inputs[indices_random] = random_words[indices_random]

  if pad:
    input_pads = tokenizer.max_len - inputs.shape[-1] # 인풋의 패딩 갯수 계산
    label_pads = tokenizer.max_len - labels.shape[-1] # 라벨의 패딩 갯수 계산

    inputs = F.pad(inputs, pad=(0, input_pads), value=tokenizer.pad_token_id)
    labels = F.pad(labels, pad=(0, label_pads), value=tokenizer.pad_token_id)

  # The rest of the time (10% of the time) we keep the masked input tokens unchanged
  return inputs, labels


def kobert_mask_tokens(tokenizer, inputs: torch.Tensor, mlm_probability=0.15, pad=True):
  """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
  """ 
  Masked Language Model을 위한 마스킹데이터 생성

  마스킹된 입력 input과
  마스킹의 정답인 label을 반환
  """
  # 라벨 생성
  labels = inputs.clone()

  # mlm_probability defaults to 0.15 in Bert
  probability_matrix = torch.full(labels.shape, mlm_probability)

  # sentencepiece 토크나이저에서
  special_tokens_mask = [
    tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
  ]
  print(special_tokens_mask)

  probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
  if tokenizer._pad_token is not None:
    padding_mask = labels.eq(tokenizer.pad_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
  masked_indices = torch.bernoulli(probability_matrix).bool()
  labels[~masked_indices] = -100  # We only compute loss on masked tokens

  # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
  indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
  inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

  # 10% of the time, we replace masked input tokens with random word
  indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
  random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
  inputs[indices_random] = random_words[indices_random]

  if pad:
    input_pads = tokenizer.max_len - inputs.shape[-1] # 인풋의 패딩 갯수 계산
    label_pads = tokenizer.max_len - labels.shape[-1] # 라벨의 패딩 갯수 계산

    inputs = F.pad(inputs, pad=(0, input_pads), value=tokenizer.pad_token_id)
    labels = F.pad(labels, pad=(0, label_pads), value=tokenizer.pad_token_id)

  # The rest of the time (10% of the time) we keep the masked input tokens unchanged
  return inputs, labels

if __name__=='__main__':
  # KoBERT 토크나이저
  tok_path = koBERTTokenizer()
  sentencepieceTokenizer = SentencepieceTokenizer(tok_path)
  sentencepieceDetokenizer = SentencepieceDetokenizer(tok_path)

  bertVocab = koBertVocab()
  bertTokenizer =BERTSPTokenizer(tok_path, bertVocab, lower=False)

  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
  test_ko_str = '오늘은 날이 매우 맑은 날로, 사람들이 밖에 나갈 준비를 하고 있다.'
  test = 'Hello, my dog is cute'

  tok = tokenizer.encode(test, max_length=tokenizer.max_len, add_special_tokens=True)
  print('tok: ',tok)
  tok = torch.tensor(tok, dtype=torch.long)

  tok1 = sentencepieceTokenizer(test_ko_str)
  tok1 =bertVocab(tok1)
  print('tok1: ',tok1)
  # print('tok1 index: ',bertVocab(tok1))

  tok1 = torch.tensor(tok1, dtype=torch.long)



  inputs, labels = mask_tokens(tokenizer,tok.unsqueeze(0), pad=True)
  # print('inputs: ',inputs)
  # print('label: ',labels)

  decode_input= tokenizer.decode(inputs.squeeze(0))
  # print('decode_input: ',decode_input)

