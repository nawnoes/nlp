import sys
sys.path.append('..')
from DLfromScratch2.common.util import preprocess, create_contexts_target, convert_one_hot

text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)

print(id_to_word)

#word2vec 학습을 위한 context와 target 데이터 준비
contexts, target = create_contexts_target(corpus,window_size=1)

print(contexts)
print(target)

#id로 바뀐 데이터에 대해 원핫 표현으로 변환
vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)