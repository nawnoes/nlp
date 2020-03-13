import DLfromScratch2.common.layers as layers
import collections

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)

        self.loss_layer = [layers.SoftmaxWithLoss() for _ in range(sample_size+1)]
        self.embed_dot_layers = [layers.EmbeddingDot(W) for _ in range(sample_size+1)]
        self.params, self.grads =[], []

        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grad += layer.grads

# 유니그램이란 하나의 연속된 단어. 2개의 연속된 단어는 바이그램. 3개의 연속된 단어는 트라이얼 그램.
# 유니그램샘플러는 한 단어를 대상으로 확율 분호를 만든다는 의
class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:  # == CPU
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0  # target이 뽑히지 않게 하기 위함
                p /= p.sum()  # 다시 정규화 해줌
                negative_sample[i, :] = np.random.choice(self.vocab_size,
                                                         size=self.sample_size,
                                                         replace=False, p=p)

        else:
            # GPU(cupy)로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size,
                                               size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample