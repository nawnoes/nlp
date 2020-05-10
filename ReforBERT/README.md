# ReforBERT
Transformer를 개선한 Reformer를 이용한 BERT. pytorch 버전

##  Introduction
2020년 트랜스포머를 개선한 리포머 발표. 
리포머는 트랜스포머의 제한 사항들을 **LSH**, **RevNet**, **Chunk**을 통해 개선하였다. 
BERT나 GPT2와 같은 큰 모델들은 많은 컴퓨팅과 메모리를 필요로 하여, 고가의 장비 없이 직접 학습시키는데 많은 제한이 있었다.
[lucidrains/reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)를 이용하여 
리포머를 이용한 **BERT**를 만들고 colab을 통해 LM을 학습해 다양한 downstream task에 이용을 목표로 한다.  
  
## Architecture
### 1. Data
한국어 위키피디아 데이터
#### Pretrain
한국어 위키 덤프
##### 전처리
  1. 위키 미러에서 덤프 다운로드
  2. sentencepiece를 이용해 8000개의 vocab 생성 
### Vocab 및 Tokenizer
SentencePiece Tokenizer 및 위키로 만든 8007개의 Vocab

### 2. Model
Reformer-pytorch의 Reformer 사용.

#### 2.1 Config


### 3. Pretrain
기본 BERT의 Masked Language Model과 Next Sentence Prediction을 사전학습에 사용.

#### 3.1 Masked Language Model

#### 3.2 Next Sentence Prediction

#### 3.3 학습

## Train Environment
Colab GPU
 

##  License
MIT

##  Author
Seonghwan Kim 

## Log
| 일자 | 내용|
|---|---|
|20.04.25| 시작 |
|20.05.08| Pretrain 코드 테스트|
|20.05.10| Colab에서 Pretrain 테스트|


# Reference
[lucidrains/reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)  
[SKTBrain/KoBERT](https://github.com/SKTBrain/KoBERT)  
[BERT(Bidirectional Encoder Representations from Transformers) 구현하기 (1/2)](https://paul-hyun.github.io/bert-01/)
