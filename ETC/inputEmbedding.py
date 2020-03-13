import sentencepiece as spm
import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
# vocab loading
vocab_file = "/Users/a60058238/Desktop/dev/workspace/nlp-study/Data/kowiki/kowiki.model"
vocab = spm.SentencePieceProcessor()
vocab.load(vocab_file)

# 입력 texts
lines = [
  "겨울은 추워요.",
  "감기 조심하세요."
]

# text를 tensor로 변환
inputs = []
for line in lines:
  pieces = vocab.encode_as_pieces(line)
  ids = vocab.encode_as_ids(line)
  inputs.append(torch.tensor(ids))
  print(pieces)

# 입력 길이가 다르므로 입력 최대 길이에 맟춰 padding(0)을 추가 해 줌
inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
# shape
print(inputs.size())
# 값
print(inputs)

n_vocab = len(vocab) # vocab count
d_hidn = 128 # hidden size
nn_emb = torch.nn.Embedding(n_vocab, d_hidn) # embedding 객체

input_embs = nn_emb(inputs) # input embedding
print(input_embs.size())

""" sinusoid position embedding """
def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table

# position encoding 구하는 절
n_seq = 64
pos_encoding = get_sinusoid_encoding_table(n_seq, d_hidn)

print (pos_encoding.shape) # 크기 출력
plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, d_hidn))
plt.ylabel('Position')
plt.colorbar()
plt.show()

pos_encoding = torch.FloatTensor(pos_encoding)
nn_pos = torch.nn.Embedding.from_pretrained(pos_encoding, freeze=True)

positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
pos_mask = inputs.eq(0)

positions.masked_fill_(pos_mask, 0)
pos_embs = nn_pos(positions) # position embedding

print(inputs)
print(positions)
print(pos_embs.size())

# Transformer에 사용될 입력
# input embedding과 postion embedding의 합으로 input_sums 생
input_sums = input_embs + pos_embs

#Scale Dot Product Attention에서 입력으로 사용될 Q, K, V
# Scale Dot Product Attention의 경우 MatMul(softmax(mask(scale(matmul(Q,K), V)
#
Q = input_sums
K = input_sums
V = input_sums
attn_mask = inputs.eq(0).unsqueeze(1).expand(Q.size(0), Q.size(1), K.size(1))
print(attn_mask.size())
print(attn_mask[0])

# softmax(Q * k^T / K-dimension) * V 수식의 Q * K-transpose 계산분
# matmul(Q,K) 부분
scores = torch.matmul(Q, K.transpose(-1, -2))
print(scores.size())
print(scores[0])

# softmax(Q * k^T / K-dimension) * V 수식의 d_head**0.5
# scale 하는 부
d_head = 64
scores = scores.mul_(1/d_head**0.5)
print(scores.size())
print(scores[0])

# Mask(opt) 하는 부분
scores.masked_fill_(attn_mask, -1e9)
print(scores.size())
print(scores[0])

#Softmax
attn_prob = nn.Softmax(dim=-1)(scores)
print(attn_prob.size())
print(attn_prob[0])

# attn_prov * V
# attn_prov는 MatMul(softmax(mask(scale(matmul(Q,K), V)에서
# softmax(mask(scale(matmul(Q,K)) 부분에 해당한다.
context = torch.matmul(attn_prob, V)
print(context.size())


W_Q = nn.Linear(d_hidn, n_head * d_head)
W_K = nn.Linear(d_hidn, n_head * d_head)
W_V = nn.Linear(d_hidn, n_head * d_head)

# (bs, n_seq, n_head * d_head)
q_s = W_Q(Q)
print(q_s.size())
# (bs, n_seq, n_head, d_head)
q_s = q_s.view(batch_size, -1, n_head, d_head)
print(q_s.size())
# (bs, n_head, n_seq, d_head)
q_s = q_s.transpose(1,2)
print(q_s.size())




