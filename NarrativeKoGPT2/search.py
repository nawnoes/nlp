import random
import torch

def beamSearch():
  None

def randomSearch(predict, k, vocab):
  # topk 중 랜덤으로 선택된 값을 반환.

  prob, indexs = torch.topk(predict, k=k, dim=-1)
  # indexs = indexs.data.view(-1)
  print('topk indexs: ', indexs[-1])

  # gen = [vocab.to_tokens(index.squeeze().tolist())[-1] for index in indexs]

  # print('topk value: ', prob)
  # print('topk count: ', index)
  # print('topk word: ', gen)
  del indexs


  return