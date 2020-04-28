# ReforBERT
리포머를 이용한 BERT 언어모델 pytorch

##  Introduction
2020년 트랜스포머를 개선한 리포머 발표. 
리포머는 트랜스포머의 제한 사항들을 **LSH**, **RevNet**, **Chunk**을 통해 개선하였다. 
BERT나 GPT2와 같은 큰 모델들은 많은 컴퓨팅과 메모리를 필요로 하여, 고가의 장비 없이 직접 학습시키는데 많은 제한이 있었다.
[lucidrains/reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)를 이용하여 
리포머를 이용한 **BERT**를 만들고 colab을 통해 LM을 학습해 다양한 downstream task에 이용을 목표로 한다.  
  
## Architecture
### Data
한국어 위키 및 나무 위키 사용 예정

### Vocab 및 Tokenizer
기존 KoBERT의 Vocab 및 SentencePiece Tokenizer 사용 예정.

#### tokenizer의 스페셜 토큰  
```python
>>> sentencepieceTokenizer.tokens[0]
Out[38]: '[UNK]'
>>> sentencepieceTokenizer.tokens[1]
Out[39]: '[PAD]'
>>> sentencepieceTokenizer.tokens[2]
Out[40]: '[CLS]'
>>> sentencepieceTokenizer.tokens[3]
Out[41]: '[SEP]'
>>> sentencepieceTokenizer.tokens[4]
Out[42]: '[MASK]'
```
### Model
Reformer-pytorch의 ReformerLmM 사용.

### Hyperparmeter

  
**KoBERT Pretrain Config**
```python
predefined_args = {
        'attention_cell': 'multi_head',
        'num_layers': 12,
        'units': 768,
        'hidden_size': 3072,
        'max_length': 512,
        'num_heads': 12,
        'scaled': True,
        'dropout': 0.1,
        'use_residual': True,
        'embed_size': 768,
        'embed_dropout': 0.1,
        'token_type_vocab_size': 2,
        'word_embed': None,
    }
```
## Train Environment
Colab GPU
 

##  License
MIT

##  Author
Seonghwan Kim 

## From
2020.04.25

# Reference
[lucidrains/reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)  
[SKTBrain/KoBERT](https://github.com/SKTBrain/KoBERT)
