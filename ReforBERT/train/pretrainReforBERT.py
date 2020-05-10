import os
import json
import logging
import numpy as np
from datetime import datetime
import math
from random import random, randrange, randint, shuffle, choice
import matplotlib.pyplot as plt
import json
import pandas as pd
from IPython.display import display
from tqdm import tqdm, tqdm_notebook, trange
import sentencepiece as spm
import wget

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import dataloader

from ReforBERT.reformer.model import ReforBertLM,ReformerLM
from ReforBERT.dataloader.kowiki import PretrainDataSet, pretrin_collate_fn
from ReforBERT.util.vocab import load_vocab
from ReforBERT.util.common import Config




""" random seed """
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


""" init_process_group """
def init_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


""" destroy_process_group """
def destroy_process_group():
    dist.destroy_process_group()

""" 모델 epoch 학습 """
def train_epoch(device, epoch, model, criterion_lm, criterion_cls, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
        for i, value in enumerate(train_loader):
            labels_cls, labels_lm, inputs, segments = map(lambda v: v.to(device), value)

            optimizer.zero_grad()
            outputs = model(inputs, segments)
            logits_cls, logits_lm = outputs[0], outputs[1]

            loss_cls = criterion_cls(logits_cls, labels_cls)
            loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))

            loss = loss_cls + loss_lm

            loss_val = loss_lm.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)



if __name__ == '__main__':
    # Data 및 Vocab 경로
    data_path = "../../Data/kowiki"
    checkpoint_path ="../checkpoint"
    vocab_path = "../../Data/kowiki/kowiki.model"

    vocab = spm.SentencePieceProcessor()
    vocab = load_vocab(vocab_path)

    print(vocab.pad_id())

    learning_rate = 5e-5 # Learning Rate
    n_epoch = 20         # Num of Epoch

    vocab_size = 8007     # vocab 크기
    max_seq_len = 512     # 최대 입력 길이
    embedding_size = 768  # 임베딩 사이
    batch_size = 128      # 학습 시 배치 크기
    device ="cpu"         # cpu or cuda

    count =10             # 데이터 분할 크기

    # pretrain 데이터 로더
    batch_size = 2#128
    dataset = PretrainDataSet(vocab, f"{data_path}/kowiki_bert_test.json")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=pretrin_collate_fn)

    # Refomer Language Model 생성
    model = ReforBertLM(
        num_tokens=vocab_size,
        dim=embedding_size,
        depth=6,
        heads=8,
        max_seq_len=max_seq_len,
        causal=True
    )

    save_pretrain = f"{checkpoint_path}/save_reforBERT_pretrain.pth"

    best_epoch, best_loss = 0, 0
    if os.path.isfile(save_pretrain):
        # best_epoch, best_loss = model.bert.load(save_pretrain)
        print(f"load pretrain from: {save_pretrain}, epoch={best_epoch}, loss={best_loss}")
        # best_epoch += 1

    model.to(device)

    criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    offset = best_epoch
    for step in range(n_epoch):
        epoch = step + offset
        if 0 < step:
            del train_loader
            dataset = PretrainDataSet(vocab, f"{data_path}/kowiki_bert_{epoch % count}.json")
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                       collate_fn=pretrin_collate_fn)
        loss = train_epoch(device, epoch, model, criterion_lm, criterion_cls, optimizer, train_loader)
        losses.append(loss)
        # model.bert.save(epoch, loss, save_pretrain)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, save_pretrain)

    # data
    data = {
        "loss": losses
    }
    df = pd.DataFrame(data)
    display(df)

    # graph
    plt.figure(figsize=[12, 4])
    plt.plot(losses, label="loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


