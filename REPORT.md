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
| 모델 | emb_dim | 128 |
| 모델 | n_heads | 4 |
| 모델 | n_layers | 4 |
| 모델 | drop_rate | 0.1 |
| 모델 | qkv_bias | False |
| 학습 | batch_size | 16 |
| 학습 | num_epochs | 30 |
| 학습 | eval_freq | 50 |
| 학습 | eval_iter | 20 |
| 최적화 | learning_rate | 3e-4 |
| 최적화 | weight_decay | 0.01 |

### 6.1.2 epoch 별 loss

| epoch | step | train_loss | val_loss | train_acc | val_acc | val_ppl |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 650 | 7.7274 | 7.7192 | 0.0085 | 0.0088 | 2251.13 |
| 2 | 1350 | 7.3395 | 7.3895 | 0.0168 | 0.0147 | 1618.90 |
| 3 | 2050 | 6.6247 | 6.7735 | 0.0429 | 0.0372 | 874.33 |
| 4 | 2750 | 6.2123 | 6.4438 | 0.0636 | 0.0543 | 628.82 |
| 5 | 3450 | 5.9857 | 6.2787 | 0.0758 | 0.0644 | 533.08 |
| 6 | 4150 | 5.8006 | 6.2019 | 0.0863 | 0.0673 | 493.71 |
| 7 | 4850 | 5.6427 | 6.1555 | 0.0952 | 0.0692 | 471.31 |
| 8 | 5550 | 5.5222 | 6.1328 | 0.0973 | 0.0667 | 460.74 |
| 9 | 6250 | 5.4302 | 6.1347 | 0.1068 | 0.0663 | 461.61 |
| 10 | 6950 | 5.3255 | 6.1415 | 0.1110 | 0.0677 | 464.73 |
| 11 | 7650 | 5.2403 | 6.1628 | 0.1183 | 0.0648 | 474.74 |
| 12 | 8350 | 5.1734 | 6.1891 | 0.1229 | 0.0640 | 487.40 |
| 13 | 9000 | 5.0728 | 6.2187 | 0.1322 | 0.0619 | 502.03 |
| 14 | 9700 | 5.0143 | 6.2636 | 0.1379 | 0.0639 | 525.09 |
| 15 | 10400 | 4.8923 | 6.3101 | 0.1462 | 0.0623 | 550.10 |
| 16 | 11100 | 4.8206 | 6.3539 | 0.1541 | 0.0619 | 574.73 |
| 17 | 11800 | 4.7704 | 6.4083 | 0.1638 | 0.0593 | 606.88 |
| 18 | 12500 | 4.6695 | 6.4511 | 0.1730 | 0.0608 | 633.37 |
| 19 | 13200 | 4.6428 | 6.5082 | 0.1753 | 0.0578 | 670.59 |
| 20 | 13900 | 4.5828 | 6.5500 | 0.1825 | 0.0584 | 699.25 |
| 21 | 14600 | 4.5494 | 6.5966 | 0.1869 | 0.0577 | 732.59 |
| 22 | 15300 | 4.4570 | 6.6496 | 0.2013 | 0.0565 | 772.46 |
| 23 | 16000 | 4.4056 | 6.6932 | 0.2089 | 0.0539 | 806.92 |
| 24 | 16700 | 4.3515 | 6.7386 | 0.2135 | 0.0542 | 844.42 |
| 25 | 17400 | 4.3093 | 6.7753 | 0.2204 | 0.0554 | 875.92 |
| 26 | 18050 | 4.2904 | 6.8208 | 0.2218 | 0.0518 | 916.73 |
| 27 | 18750 | 4.2319 | 6.8889 | 0.2292 | 0.0526 | 981.28 |
| 28 | 19450 | 4.1824 | 6.9509 | 0.2346 | 0.0517 | 1044.13 |
| 29 | 20150 | 4.1325 | 7.0021 | 0.2461 | 0.0491 | 1098.96 |
| 30 | 20850 | 4.0704 | 7.0678 | 0.2504 | 0.0500 | 1173.53 |

### 6.1.3 결과

| 항목 | 내용 |
| --- | --- |
| final train loss |  |
| final validation loss |  |
| best validation loss |  |
| best epoch |  |
| checkpoint 경로 |  |

#### 손실 그래프

<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/68e21829-9c81-440c-b06f-fe004a67968c" />

- x축: epoch
- y축: train loss, validation loss
- 그래프 아래에 loss 변화 경향을 1~2문장으로 요약

#### 정확도 그래프

<img width="700" height="470" alt="image" src="https://github.com/user-attachments/assets/a6d9e711-d1e4-42bf-b9d1-99200f4a1391" />

- x축: epoch
- y축: train accuracy, validation accuracy
- 그래프 아래에 accuracy 변화 경향을 1~2문장으로 요약

#### 생성 샘플

| epoch | 생성 샘플 |
|---:|---|
| 1 | 영화고이기이은 영화가.을을도지..하고만의고한이도. .도과가가도만가은,을..로한도이..가 을로이영화에...은아과.가이가은을의는아..고에 에 영화다..이다 가하고만지가가은를. |
| 2 | 영화다 !! 이 4리. 그은 시간의 좋았어서 우이 된음 에 하고 지루을 하다. 나를 더 좋다. 한국을 일도 어지하수은 정말 어�가...... 3에 한기...? 오로 진짜 있다. 이런 4 어가는 이이 이를 볼, 보긴다.. 한 시도 보고 말을 왜 |
| 3 | 영화! 이하는 사람에 영화~!!?? fegueh Alidasestobciliasteittiethot hevestiesteelos ht etespd wlnot fhee |
| 4 | 영화인데. 그래도 저리지만 더불어 이뻐 마지막, 김상하는 거짓을 남는 건지훈을 쏭제에서 봤는데 이런걸만하는 걸리기싫어졌을까 제, ㄱㄱ♥♥!! 우시카나, 어쨌는 장면의영화만도 아까운 영화 명작.444.35년전작. "볼수있 |
| 5 | 영화임새가있다. 영화보던거지로 빌려온다. 아들이 좀 더 재밌지만... 내..이건 뭐다 이거보고있다 그것에 보지면서 봤어야 했다 그리다살다 개발만하는 거 같아직도 못해가능한장면도 많이 준다 뭐니 기대없이 웃게되는 영화! 너무 재밌어놨는데 정말 재밌는데 |

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
