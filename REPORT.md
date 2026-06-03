# mini GPT 구현 과제 보고서

## 0. 반·팀원

| 항목 | 내용 |
| --- | --- |
| 반 | SW_AI LAB 302반 |
| 팀명 | 6팀 |
| 팀원 | 송채강, 이규정, 이원재 |

---

## 1. 구현 현황

| 단계 | 구현 내용 | 구현 파일 | 담당자 |
| --- | --- | --- | --- |
| 1 | UTF-8 byte-level BPE tokenizer | `src/bpe.py` |  |
| 2 | GPTDataset, create_dataloader, InputEmbedding | `src/dataset.py`, `src/embeddings.py` |  |
| 3 | MultiHeadAttention, causal mask | `src/attention.py` |  |
| 4 | LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, generate_text_simple | `src/model.py` |  |
| 5 | loss 계산, checkpoint, generate, train_model | `src/train.py` |  |
| 6 | NSMC 감성 분류 Dataset과 classifier | `src/finetune.py` |  |

---

## 2. 테스트 통과 현황

| 실행 명령 | 결과 | 비고 |
| --- | --- | --- |
| `pytest tests/test_bpe.py -v` | 통과 | 반복 corpus에서 가장 자주 등장하는 pair가 merges[0]에 들어가는지 확인 추가 |
| `pytest tests/test_dataset.py -v` | 통과 |  |
| `pytest tests/test_attention.py -v` | 통과 | attention weight가 올바르게 정규화되는지는 확인 테스트 추가 |
| `pytest tests/test_model.py -v` | 통과 |  |
| `pytest tests/test_train.py -v` | 통과 |  |
| `pytest tests/test_finetune.py -v` | 통과 | 짧은 입력은 padding되고, 긴 입력은 truncation되는지 확인하는 테스트 추가 |
| `pytest tests/ -v` | 통과 |  |

---

## 3. 데이터

| 항목 | 내용 |
| --- | --- |
| 원본 데이터 | NSMC |
| 원본 경로 | `data/ratings_train.txt`, `data/ratings_test.txt` |
| 사전 학습 데이터 | `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt` |
| 미세 조정 데이터 | `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl` |
| 전처리 방식 | 빈 리뷰 제거, 공백 정리, train/validation 분리 |
| 사용한 데이터 크기 | 사전 학습: train 1,379,486자, val 120,560자 |

---

## 4. BPE

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/bpe.py` |
| BPE 방식 | UTF-8 byte-level BPE |
| 인코딩 방식 | 문장 단위 `<bos>`/`<eos>` 추가 |
| 특수 토큰 ID | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |
| byte token ID 범위 | 4~259 |
| vocab_size | 3000 |
| 학습 corpus 크기 | corpus[:1,500,000] / 실제 사용 1,379,486자 |
| 어휘 학습 시간 | 17.01분 (1020.73초) |
| vocabulary 저장 경로 | data\nsmc_bpe_vocab_3000_timed.json |
| 인코딩/디코딩 복원 예시 | decode(encode("이 영화는 좋았다")) == "이 영화는 좋았다" → True |

---

## 5. 모델 구조

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/model.py` |
| 전체 구조 | InputEmbedding -> 4 x TransformerBlock -> LayerNorm -> LM head |
| vocab_size | 3000 |
| context_length | 128 |
| emb_dim | 256 |
| n_heads | 4 |
| n_layers | 4 |
| drop_rate | 0.15 |
| qkv_bias | False |
| stride | 64 |
| learning_rate | 2e-4 |
| weight_decay | 0.03 |
| eval_freq | 200 |
| eval_iter | 10 |
| 총 파라미터 수 | 4,725,248 |

---

## 6. 사전 학습

### 6.1.1 1차 시도

| 구분 | 항목 | 값 |
| --- | --- | --- |
| 구현 | 구현 파일 | `src/train.py` |
| 모델 | vocab_size | 3000 |
| 모델 | context_length | 128 |
| 모델 | stride | 64 |
| 모델 | emb_dim | 256 |
| 모델 | n_heads | 4 |
| 모델 | n_layers | 4 |
| 모델 | drop_rate | 0.15 |
| 모델 | qkv_bias | False |
| 학습 | batch_size | 4 |
| 학습 | num_epochs | 100 |
| 학습 | eval_freq | 100 |
| 학습 | eval_iter | 20 |
| 최적화 | learning_rate | 3e-4 |
| 최적화 | weight_decay | 0.01 |

### 6.1.2 결과

| 항목 | 내용 |
| --- | --- |
| final train loss |  |
| final validation loss |  |
| best validation loss |  |
| best epoch |  |
| checkpoint 경로 |  |

#### 손실 그래프

![사전학습 손실 그래프](results/pretrain_loss_curve.png)

- x축: epoch
- y축: train loss, validation loss
- 그래프 아래에 loss 변화 경향을 1~2문장으로 요약

#### 정확도 그래프

![사전학습 정확도 그래프](results/pretrain_accuracy_curve.png)

- x축: epoch
- y축: train accuracy, validation accuracy
- 그래프 아래에 accuracy 변화 경향을 1~2문장으로 요약

#### 생성 샘플

| epoch | 생성 결과 |
| --- | --- |
| 1 |  |
| 2 |  |
| 3 |  |

#### 결과 해석

- validation loss가 가장 낮았던 epoch와 그 이후 추세를 간단히 정리
- 생성 결과가 학습 초반 대비 얼마나 자연스러워졌는지 한두 문장으로 설명

---

## 7. 미세 조정

### 7.1.1 설정

| 구분 | 항목 | 값 |
| --- | --- | --- |
| 구현 | 구현 파일 | `src/finetune.py` |
| 구현 | 과제 | NSMC 리뷰 긍정/부정 분류 |
| 데이터 | 데이터 포맷 | JSONL, `text`, `label` |
| 모델 | tokenizer_vocab_size | 3000 |
| 모델 | max_length | 128 |
| 모델 | emb_dim | 256 |
| 모델 | n_heads | 4 |
| 모델 | n_layers | 4 |
| 모델 | drop_rate | 0.2 |
| 모델 | qkv_bias | False |
| 분류기 | classifier_drop_rate | 0.2 |
| 학습 | batch_size | 128 |
| 학습 | num_epochs | 8 |
| 최적화 | learning_rate | 2e-5 |
| 최적화 | weight_decay | 0.02 |

### 7.1.2 결과

| 항목 | 내용 |
| --- | --- |
| 구현 파일 | `src/finetune.py` |
| 과제 | NSMC 리뷰 긍정/부정 분류 |
| 데이터 포맷 | JSONL, `text`, `label` |
| max_length | 128 |
| batch_size | 128 |
| backbone learning rate | 2e-5 |
| classifier learning rate | 2e-5 |
| validation loss / accuracy | 0.4169 / 0.8218 |
| test loss / accuracy | 0.4146 / 0.8274 |

|  | **epoch** | **train_loss** | **val_loss** | **train_acc** | **val_acc** |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0.662586 | 0.509556 | 0.639062 | 0.748146 |
| 1 | 2 | 0.549443 | 0.482957 | 0.722108 | 0.770231 |
| 2 | 3 | 0.503963 | 0.462987 | 0.754044 | 0.781398 |
| 3 | 4 | 0.475936 | 0.452796 | 0.771783 | 0.794066 |
| 4 | 5 | 0.456949 | 0.436055 | 0.784001 | 0.805734 |
| 5 | 6 | 0.440778 | 0.445127 | 0.793740 | 0.800150 |
| 6 | 7 | 0.427146 | 0.418713 | 0.801364 | 0.816735 |
| 7 | 8 | 0.415599 | 0.442509 | 0.807103 | 0.816068 |
| 8 | 9 | 0.407880 | 0.433760 | 0.812589 | 0.817318 |
| 9 | 10 | 0.398865 | 0.416875 | 0.817401 | 0.821818 |
| 10 | 11 | 0.389004 | 0.416571 | 0.824009 | 0.823985 |

#### 손실 그래프

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/79addd18-6aba-4b9f-acef-b3aa08edd213" />

- Fine-tuning 결과, train loss와 validation loss가 모두 전반적으로 감소했다.
- validation loss는 일부 epoch에서 작은 흔들림이 있었지만 전체적으로 하락 추세를 유지했다.

#### 정확도 그래프

<img width="708" height="470" alt="image" src="https://github.com/user-attachments/assets/19ae2319-938f-4324-8206-e87d2489b013" />

- train accuracy와 validation accuracy도 함께 상승하였다.
- validation accuracy도 약 0.825까지 상승하였다.

#### 오류 예시

| text | pred_label | negative_prob | positive_prob |
| --- | --- | ---: | ---: |
| 이 영화는 정말 좋았다 | 긍정 | 0.007649 | 0.992351 |
| 시간이 너무 아까웠다 | 부정 | 0.989832 | 0.010168 |
| 배우 연기는 좋았지만 이야기는 지루했다 | 부정 | 0.906879 | 0.093121 |
| 기대보다 훨씬 재미있었다 | 긍정 | 0.021235 | 0.978765 |
| 다시는 보고 싶지 않은 영화였다 | 긍정 | 0.038633 | 0.961367 |

#### 결과 해석

- best validation loss를 기준으로 validation loss가 가장 낮았던 epoch의 checkpoint를 최종 모델 기준으로 삼았다. 마지막 epoch 모델은 train loss가 더 낮더라도 validation 성능이 정체되거나 악화될 수 있으므로 best checkpoint를 선택하였다.

- test accuracy는 validation accuracy와 큰 차이 없이 비슷한 수준을 보여, validation에서의 성능이 test data에서도 대체로 유지되었다고 볼 수 있다. 이는 모델이 특정 validation에만 과하게 맞춰진 것은 아니라는 뜻이 된다.

- 오분류 예시인 “다시는 보고 싶지 않은 영화였다”에서는 부정 표현 “다시는 ~지 않은”보다 “보고 싶다”라는 긍정 표현에 더 강하게 반응한 것으로 보인다. 이는 짧은 리뷰 안에서도 부정/반전 표현을 정확히 해석하는 데 한계가 있음을 보여준다.

---

## 8. 실험 환경

| 항목 | 내용 |
| --- | --- |
| Python | (예: Python 3.11) |
| PyTorch | (예: PyTorch 2.x) |
| 실행 환경 | Colab GPU / Colab CPU / 로컬 |
| GPU/CPU 정보 |  |
| 총 학습 소요 시간 |  |

---

## 9. 고찰

- 어려웠던 점
- 한국어 byte-level BPE 구현에서 조심한 점
- loss가 줄어든 이유 또는 줄어들지 않은 이유
- 과적합·과소적합 여부
- 하이퍼파라미터 변경 시도와 결과
- 다음에 개선하고 싶은 점
