#Emotion Analization from Naver movie review
import Transformer.modules.Transformer as Transformer
import torch.nn as nn
import torch
import tqdm
import json
import numpy as np


class MovieClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = Transformer(self.config)
        self.projection = nn.Linear(self.config.d_hidn, self.config.n_output, bias = False)

    def forward(self, enc_input, dec_inputs):
        # (bs, n_dec_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.transformer(enc_input, dec_inputs)

        # (bs, d_hidn)
        dec_outputs , _ = torch.max(dec_outputs, dim=1)
        # (bs, n_output)
        logits = self.projection(dec_outputs)

        # (bs, n_output), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        return logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs

    def save(self, epoch, loss, score, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "score": score,
            "state_dict": self.state_dict()
        }, path)

    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"], save["score"]
"""영화 분류 데이터 셋"""
class MovieDataSet(torch.utils.data.Dataset):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.labels = []
        self.sentences =[]

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt +=1
        with open(infile,"r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc = f":oading {infile}", unit="lines")):
                data = json.loads(line)
                #입력파일로부터 label을 읽는다
                self.labels.append(data["label"])
                #입력 파일로부터 doc token을 읽어 숫자 id 로 변경한다.
                self.sentences.append([vocab.piece_to_id(p) for p in data["doc"]])

    def __len__(self):
        assert len(self.labels) == len(self.sentences)
        return len(self.labels)

    def __getitem__(self, item):
        return (torch.tensor(self.labels[item]),
                torch.tensor(self.sentences[item]),
                #디코더의 입력은 '[BOS]'로 고정
               torch.tensor([self.vocab.piece_to_id("[BOS]")]))

"""movie data collate_fn"""
def movie_collate_fn(inputs):
    labels, enc_inputs, dec_inputs = list(zip(*inputs))

    #인코더 인풋의 길이가 같아지도록 짧은 문장에서는 0패딩을 추가
    enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first= True, padding_value = 0)
    #디코더 인풋의 길이가 같아지도록 짧은 문장에 대해서는 0 패딩 추가
    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first= True, padding_value = 0)

    batch =[
        torch.stack(labels, dim=0),
        enc_inputs,
        dec_inputs,
    ]
    return batch

"""DataLoader"""
#위에서 정의한 DataSet과 collate_fn을 이용해 학습용과 평가용 DataLoader를 만든다.
def dataLoader(vocab):
    batch_size =128

    train_dataset = MovieDataSet(vocab, "<path of data>/ratings_train.json")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn = movie_collate_fn)

    test_dataset = MovieDataSet(vocab, "<path of data>/ratings_test.json")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=movie_collate_fn)

    return train_loader, test_loader

""" 데이터 로더 """
def build_data_loader(vocab, infile, args, shuffle=True):
    dataset = MovieDataSet(vocab, infile)
    if 1 < args.n_gpu and shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, sampler=sampler, collate_fn=movie_collate_fn)
    else:
        sampler = None
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, sampler=sampler, shuffle=shuffle, collate_fn=movie_collate_fn)
    return loader, sampler

"""모델 epoch 평가"""
def eval_epoch(config, model, data_loader):
    matchs =[]
    model.eval()

    n_word_total = 0
    n_correct_total = 0

    with tqdm(total=len(data_loader), desc=f"Valid") as pbar:
        for i, value in enumerate(data_loader):

            labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

            # 1. 인코더, 디코더 입력으로 movieClassification을 실행
            outputs =model(enc_inputs, dec_inputs)
            # 2. 첫번째 값이 예측 logits 값
            logits = outputs[0]
            # 3. logit의 최대값 index를 구한다 .
            _, indices = logits.max(1)

            # 4. 3에서 구한 값과 labels의 값이 같은지 비교
            match = torch.eq(indices, labels).detach()
            matchs.extend(match.cpu())
            accuracy = np.sum(matchs)/ len(matchs) if 0 < len(matchs) else 0

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}")


    return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0


""" 모델 epoch 학습"""
def train_epoch(config, epoch, model, criterion, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm(total = len(train_loader), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            labels, enc_inputs, dec_inputs = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()
            # 1.인코더 인풋, 디코더 인풋을 입력으로 MovieClassification을 실행
            outputs = model(enc_inputs, dec_inputs)
            # 2. 1번의 결과 중 첫번째 값이 예측 logits
            logits = outputs[0]

            # 3. logits 값과 labels의 값을 이용해 Loss를 계산
            loss = criterion(logits, labels)
            loss_val = loss.item()
            losses.append(loss_val)

            # 4. loss와 optimizer를 이용해 학습
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss:{loss_val:.3f} ({np.mean(losses):.3f})")

    return np.mean(losses)


