"""
GPT-2 Pretraining 테스트 코드

self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 부분에서
트랜스포머의 결과 (토큰수, 임베딩 수) * (임베딩 수 * 단어수)

"""
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=input_ids)
loss, logits = outputs[:2]

print('loss {}' . format(loss))
print('logits {}' . format(logits))