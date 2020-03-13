import sys
sys.path.append('..')
import numpy as np
from .functions import softmax, cross_entropy_error

class Embedding:
    def __init__(self, W):
        self.params =[W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self,idx):
        W, = self.params
        self.idx = idx
        out = W[idx]

        return out

    # 임베딩 계층의 순전파는 가중치 W의 특정행을 추출한다.
    # 역전파에서도 앞에서 전해진 기울기를 다음 층으로 흘려주기만 하면 된다.
    def backward(self, dout):
        dW, =self.grads # 가중치 기울기를 꺼내고
        dW[...] = 0 # dW 원소를 0으로 덮어 쓴다. 형상은 유지 & 원소들은 0

        #dW[self.idx] = dout # 앞에서 전해진 기울기를 idx번째 행에 할당. # 안좋은 예, idx가 중복인 경우에 문제 발생
        for i, word_id in enumerate(self.idx):
            dW[word_id] += dout[i]
        #혹은
        # np.add.at(A, idx, B)는 B를 A의 idx 번째 행에 더해준다.
        # np.add.at(dW, self.idx, dout)
        return None

class EmbeddingDot:
    def __init__(self, W):
        self.embed= Embedding(W) # 임베딩 계층
        self.params = self.embed.params # 매개변수를 저
        self.grads = self.embed.grads # 기울기를 저장
        self.cache = None # 순전파 시 계산 결과를 잠시 저

    """
    - 인수로 은닉층 뉴런 h, 단어 id의 넘파일 배열(idx)을 받는다.
    - idx는 단어 id의 배열.
    - 배열로 받는 이유는 데이터를 한꺼번에 처리하는 미니배치 처리를 가정했기 때문
    """
    def forward(self, h, idx):
        target_W = self.embed.forward(idx) # 먼저 Embedding 계층의 forward(idx)를 호출

        # 내적 계산. 각 자리의 요소들끼리 곱
        # 넘파이 배열의 * 연산은 원소별 곱을 수행
        # axis =1 결과를 행마다 더한다
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache

        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout *h
        self.embed.backward(dtarget_W)
        dh = dout * target_W

        return dh




class MatMul:
    def __init__(self, W):
        # 학습하는 매개변수를 params에 보관
        # 대응하는 형태로, 기울기는 grads에 보관
        # 역전파에서는 dx와 dW를 구해, 가중치의 기울기를 인스턴스 변수인 grads에 저
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x =None

    def forword(self,x) :
        W, = self.params
        out = np.matmul(x,W)
        self.x = x

        return out

    def backward(self,dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)

        # 깊은 복사
        self.grads[0][...] = dW # 생략기호 ...사용. 생략 기호는 넘파이의 덮어쓰기를 수행

        return dx

class Sigmod:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out=out
        return out
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

class Affine:
    def __init__(self,W,b):
        self.params =[W,b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x =None

    def forward(self, x):
        W,b= self.params
        out = np.matmul(x,W)+b
        self.x =x
        return out
    def backward(self, dout):
        W,b = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis =0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid의 출력
        self.t = None  # 정답 데이터

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        # cross_entropy_error()에 이미 있어서 굳이 필요 없을 듯
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx /= batch_size

        return dx

