import torch
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
import random

tok_path = get_tokenizer()
model, vocab = get_pytorch_kogpt2_model()
tok = SentencepieceTokenizer(tok_path)
while 1:

  sent = input('문장을 입력해주세요: ')
  toked = tok(sent)
  count = 0
  input_size = 150

  while 1:
    input_ids = torch.tensor([vocab[vocab.bos_token],]  + vocab[toked]).unsqueeze(0)
    predicts = model(input_ids)
    pred = predicts[0]
    print('predicts:', torch.argmax(pred, axis=-1).squeeze())
    gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
    if gen == '</s>':
    #   # break
    #   continue
      indx = random.randrange(0,7)
      # pred = predicts[indx]
      print('to_tokens:',vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist()))
      # gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]

    if count >= input_size:
      break
    # print('gen '+str(count)+': ',gen)
    sent += gen.replace('▁', ' ')
    toked = tok(sent)

    if count%30==0:
      sent += '\n'
      print(sent)

    count += 1
  print(sent)