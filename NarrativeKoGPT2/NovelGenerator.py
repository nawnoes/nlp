import random
import torch
from torch.utils.data import DataLoader # 데이터로더
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from kogpt2.utils import download, tokenizer
from model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
import gluonnlp
import search

### 1. koGPT2 Config
ctx= 'cpu'#'cuda' #'cpu' #학습 Device CPU or GPU. colab의 경우 GPU 사용
cachedir='~/kogpt2/' # KoGPT-2 모델 다운로드 경로
epoch =200  # 학습 epoch
save_path = './checkpoint'
load_path = 'checkpoint/checkpoint_0_epoch_134.tar'
load_path_fairy_tale = 'checkpoint/fairy_tale_checkpoint_epoch_98.tar'

#use_cuda = True # Colab내 GPU 사용을 위한 값

pytorch_kogpt2 = {
    'url':
    'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}
kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000
}

### 2. Vocab 불러오기
# download vocab
vocab_info = tokenizer
vocab_path = download(vocab_info['url'],
                       vocab_info['fname'],
                       vocab_info['chksum'],
                       cachedir=cachedir)

### 3. 체크포인트 및 디바이스 설정
# Device 설정
device = torch.device(ctx)
# 저장한 Checkpoint 불러오기
checkpoint = torch.load(load_path_fairy_tale, map_location=device)

# KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
kogpt2model.load_state_dict(checkpoint['model_state_dict'])

kogpt2model.eval()
vocab_b_obj = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                     mask_token=None,
                                                     sep_token=None,
                                                     cls_token=None,
                                                     unknown_token='<unk>',
                                                     padding_token='<pad>',
                                                     bos_token='<s>',
                                                     eos_token='</s>')
### 4. Tokenizer
tok_path = get_tokenizer()
model, vocab = kogpt2model, vocab_b_obj
tok = SentencepieceTokenizer(tok_path)

### 5. Text Generation
sent =''
while 1:

  tmp_sent = input('다음...: ')
  sent = sent+tmp_sent

  toked = tok(sent)
  count = 0
  generated_text =''
  input_size = 500

  if len(toked) >1022:
    break

  while 1:
    input_ids = torch.tensor([vocab[vocab.bos_token],]  + vocab[toked]).unsqueeze(0)
    predicts = model(input_ids)
    pred = predicts[0]
    # print('predicts:', torch.argmax(pred, axis=-1).squeeze())
    gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]

    search.randomSearch(pred, 5, vocab)
    if gen == '</s>':
      print('to_tokens:',vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist()))
    if count>input_size:
      sent += gen.replace('▁', ' ')
      generated_text += gen.replace('▁', ' ')
      sent += '\n'
      generated_text += '\n'
      toked = tok(sent)
      count =0
      break
      # print('to_tokens:',vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist()))
    # if count >= input_size:
    #   break
    sent += gen.replace('▁', ' ')
    generated_text += gen.replace('▁', ' ')
    # print(generated_text)

    toked = tok(sent)
    count += 1
    print(count)
  print(generated_text)
  generated_text=''
print(sent)