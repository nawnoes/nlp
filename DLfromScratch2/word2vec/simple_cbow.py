import numpy as np
from DLfromScratch2.common.layers import MatMul,SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V,H = vocab_size, hidden_size

        #가중치 초기화
        # 2개의 가중치 생성
        # 무작위로 초기화
        # 넘파이 배열 타입을 astype('f')로 지정. 32비트 부동소수점 수로 초기
        W_in = 0.01 * np.random.randn(V,H).astype('f')
        W_out = 0.01 * np.random.randn(H,V).astype('f')

        # 계층 생성
        # context를 처리하는 MatMul 계층은 context에서 사용하는 단어 수(윈도우 크기) 만큼 만든다.
        # 그리고 입력 MatMul 계층들은 모두 같은 가중치를 갖도록 한다.
        self.in_layer0=MatMul(W_in)
        self.in_layer1=MatMul(W_in)
        self.out_layer=MatMul(W_out)
        self.loss_layer=SoftmaxWithLoss()

        #모든 가중치와 기울기를 리스트에 모은다
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [],[]

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        #인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs=W_in

    #contexts는 3차원 넘파이 배열로 가정 (미니배치수, 맥락의 윈도우크기, 원핫벡터)
    def forward(self, contexts, target):
        h0 = self.in_layer0.forword(contexts[:,0])
        h1 = self.in_layer1.forword(contexts[:,1])
        h = (h0+h1) *0.5

        score=self.out_layer.forword(h)
        loss=self.loss_layer.forward(score,target)

        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5

        self.in_layer1.backward(da)
        self.in_layer0.backward(da)

        return None